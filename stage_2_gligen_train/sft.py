import torch
import torch.nn.functional as F
import lightning as L

import fnmatch
import json
import math
import os
import shutil
from typing import List, Optional
from streaming import StreamingDataset

import numpy as np
import torch
import torch.utils.checkpoint
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.optimization import get_scheduler
from safetensors.torch import save_file
from tqdm.auto import tqdm

from dataset_and_utils import (
    TokenEmbeddingsHandler,
    load_models,
    unet_attn_processors_state_dict,
)
import streaming
from streaming.base.format.mds.encodings import Encoding, _encodings
import numpy as np
from typing import Any
import torch


class np32(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.float32)


_encodings["np32"] = np32


import os
import shutil


local_train_dir = "./local_train_dir"
if os.path.exists(local_train_dir):
    shutil.rmtree(local_train_dir)

local_val_dir = "./local_val_dir"
if os.path.exists(local_val_dir):
    shutil.rmtree(local_val_dir)


streaming.base.util.clean_stale_shared_memory()


def main(
    pretrained_model_name_or_path: Optional[
        str
    ] = "stabilityai/stable-diffusion-xl-base-1.0",
    revision: Optional[str] = None,
    output_dir: str = "./checkpoints/patch_pool",
    seed: Optional[int] = 42,
    resolution: int = 512,
    crops_coords_top_left_h: int = 0,
    crops_coords_top_left_w: int = 0,
    train_batch_size: int = 8,
    do_cache: bool = True,
    num_train_epochs: int = 600,
    max_train_steps: Optional[int] = None,
    valid_steps: int = 200,  # default to no checkpoints
    checkpoint_steps: int = 2000,
    gradient_accumulation_steps: int = 1,  # todo
    unet_learning_rate: float = 4e-5,
    ti_lr: float = 3e-4,
    pivot_halfway: bool = True,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 500,
    lr_num_cycles: int = 1,
    lr_power: float = 1.0,
    dataloader_num_workers: int = 0,
    max_grad_norm: float = 1.0,  # todo with tests
    allow_tf32: bool = True,
    mixed_precision: Optional[str] = "bf16",
    device: str = "cuda:0",
    token_dict: dict = {"TOKEN": "<s0>"},
    inserting_list_tokens: List[str] = ["<s0>", "<s1>"],
    verbose: bool = True,
    remote_train_dir: str = "./dataset",
    remote_val_dir: str = "./dataset_val",
) -> None:
    import wandb

    wandb.init(project="Vendor", entity="simo", name=output_dir.split("/")[-1])

    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if not seed:
        seed = np.random.randint(0, 2**32 - 1)
    print("Using seed", seed)
    torch.manual_seed(seed)

    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    fabric = L.Fabric(accelerator="cuda", devices=8, precision="bf16-mixed")
    fabric.launch()

    if scale_lr:
        unet_learning_rate = (
            unet_learning_rate * gradient_accumulation_steps * train_batch_size
        )

    (
        tok1,
        tok2,
        noise_scheduler,
        text_encoder_one,
        text_encoder_two,
        _,
        unet,
    ) = load_models(pretrained_model_name_or_path, revision, device, weight_dtype)

    print("# PTI : Loaded models")

    # Initialize new tokens for training.

    embedding_handler = TokenEmbeddingsHandler(
        [text_encoder_one, text_encoder_two], [tok1, tok2]
    )
    embedding_handler.initialize_new_tokens(inserting_toks=inserting_list_tokens)

    text_encoders = [text_encoder_one, text_encoder_two]

    unet_param_to_optimize = []
    # fine tune only attn weights

    text_encoder_parameters = []
    for text_encoder in text_encoders:
        for name, param in text_encoder.named_parameters():
            if "token_embedding" in name:
                param.requires_grad = True
                print(name)
                text_encoder_parameters.append(param)
            else:
                param.requires_grad = False

    WHITELIST_PATTERNS = [
        "*.attn*.weight",
        #"*.temporal*",
    ]  # TODO : make this a parameter
    BLACKLIST_PATTERNS = ["*time*"]

    unet_param_to_optimize_names = []
    for name, param in unet.named_parameters():
        if any(
            fnmatch.fnmatch(name, pattern) for pattern in WHITELIST_PATTERNS
        ) and not any(fnmatch.fnmatch(name, pattern) for pattern in BLACKLIST_PATTERNS):
            param.requires_grad_(True)
            unet_param_to_optimize_names.append(name)
            unet_param_to_optimize.append(param)
            print(f"Training: {name}")
        else:
            param.requires_grad_(False)

    # Optimizer creation
    params_to_optimize = [
        {
            "params": unet_param_to_optimize,
            "lr": unet_learning_rate,
        },
        {
            "params": text_encoder_parameters,
            "lr": ti_lr,
            "weight_decay": 1e-3,
        },
    ]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        weight_decay=1e-4,
    )

    # Remote directory (S3 or local filesystem) where dataset is stored

    # Local directory where dataset is cached during operation

    local_train_dir = "/root/bigdisk/project_structured_prompt/stage_2_gligen_train/grit_mds"

    train_dataset = StreamingDataset(
        local=local_train_dir,
        remote=remote_train_dir,
        split=None,
        shuffle=True,
        shuffle_algo="naive",
        num_canonical_nodes=1,
    )

    local_val_dir = "/root/bigdisk/project_structured_prompt/stage_2_gligen_train/grit_mds_test"

    val_dataset = StreamingDataset(
        local=local_val_dir,
        remote=remote_val_dir,
        split=None,
        shuffle=False,
        num_canonical_nodes=1,
    )

    print("# PTI : Loaded dataset")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=dataloader_num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_batch_size,
        num_workers=dataloader_num_workers,
    )

    unet, optimizer = fabric.setup(unet, optimizer)
    text_encoder_one = fabric.to_device(text_encoder_one).to(weight_dtype)
    text_encoder_two = fabric.to_device(text_encoder_two).to(weight_dtype)

    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = train_batch_size * gradient_accumulation_steps

    if verbose:
        print(f"# PTI :  Running training ")
        print(f"# PTI :  Num examples = {len(train_dataset)}")
        print(f"# PTI :  Num batches each epoch = {len(train_dataloader)}")
        print(f"# PTI :  Num Epochs = {num_train_epochs}")
        print(f"# PTI :  Instantaneous batch size per device = {train_batch_size}")
        print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        print(f"# PTI :  Gradient Accumulation steps = {gradient_accumulation_steps}")
        print(f"# PTI :  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps))

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    valid_prompts = [
        "epic castle landscape",
        "A bigfoot walking in the snowstorm",
        "A squirrel eating a burger",
        "An astronaut flying in space, 4k, high resolution.",
    ]

    wandb.watch(unet.module)
    wandb.watch(text_encoder_one)
    wandb.watch(text_encoder_two)

    for epoch in range(first_epoch, num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            progress_bar.update(1)
            progress_bar.set_description(f"# PTI :step: {global_step}, epoch: {epoch}")

            sts = batch["caption_output"]
            vae_latent = batch["vae_output"].reshape(-1, 4, 80, 80) * 0.13025

            # tokens to text embeds
            prompt_embeds_list = []
            for tokenizer, text_encoder in zip((tok1, tok2), text_encoders):
                tok = tokenizer(
                    sts,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).input_ids

                prompt_embeds_out = text_encoder(
                    tok.to(text_encoder.device),
                    output_hidden_states=True,
                )

                pooled_prompt_embeds = prompt_embeds_out[0]
                prompt_embeds = prompt_embeds_out.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

            # Create Spatial-dimensional conditions.

            original_size = (resolution, resolution)
            target_size = (resolution, resolution)
            crops_coords_top_left = (crops_coords_top_left_h, crops_coords_top_left_w)
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])

            add_time_ids = add_time_ids.to(device, dtype=prompt_embeds.dtype).repeat(
                bs_embed, 1
            )

            added_kw = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(vae_latent)
            bsz = vae_latent.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=vae_latent.device,
            )
            timesteps = timesteps.long()

            noisy_model_input = noise_scheduler.add_noise(vae_latent, noise, timesteps)

            # Predict the noise residual
            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=added_kw,
            )[0]

            loss = (model_pred - noise).pow(2)
            loss = loss.mean()

            fabric.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            wandb.log({"train_loss": loss.item()})

            # every step, we reset the embeddings to the original embeddings.

            for idx, text_encoder in enumerate(text_encoders):
                embedding_handler.retract_embeddings()

            if global_step % valid_steps == 0:
                

                if fabric.global_rank == 0:
                    if global_step % checkpoint_steps == 0 and global_step > 0:
                        os.makedirs(f"{output_dir}/{global_step}", exist_ok=True)
                        unet_path = f"{output_dir}/{global_step}/unet.pth"
                        text_encoder_one_path = f"{output_dir}/{global_step}/text_encoder_one.pth"
                        text_encoder_two_path = f"{output_dir}/{global_step}/text_encoder_two.pth"

                        # save all.
                        save_file(unet.state_dict(), unet_path)
                        save_file(text_encoder_one.state_dict(), text_encoder_one_path)
                        save_file(text_encoder_two.state_dict(), text_encoder_two_path)

                        
                    from diffusers import DiffusionPipeline

                    pipe = DiffusionPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        unet=unet.module,
                        torch_dtype=torch.bfloat16,
                        use_safetensors=True,
                        variant="fp16",
                    ).to(fabric.device)
                    generator = torch.Generator()
                    generator.manual_seed(step)
                    images = pipe(
                        valid_prompts,
                        guidance_scale=5.0,
                        generator=generator,
                        width=640,
                        height=640,
                    ).images
                    log_images = []
                    for img, promp in zip(images, valid_prompts):
                        _img = wandb.Image(img, caption=promp)
                        wandb.log({"images": _img})

                    del pipe

                # Validation Loss
                with torch.no_grad():
                    val_loss = 0.0
                    tot_n = 0.0
                    for step, batch in enumerate(val_dataloader):
                        sts = batch["caption_output"]
                       
                        vae_latent = (
                            batch["vae_output"].reshape(-1, 4, 80, 80) * 0.13025
                        )

                        # tokens to text embeds
                        prompt_embeds_list = []
                        for tokenizer, text_encoder in zip((tok1, tok2), text_encoders):
                            tok = tokenizer(
                                sts,
                                padding="max_length",
                                max_length=77,
                                truncation=True,
                                add_special_tokens=True,
                                return_tensors="pt",
                            ).input_ids

                            prompt_embeds_out = text_encoder(
                                tok.to(text_encoder.device),
                                output_hidden_states=True,
                            )

                            pooled_prompt_embeds = prompt_embeds_out[0]
                            prompt_embeds = prompt_embeds_out.hidden_states[-2]
                            bs_embed, seq_len, _ = prompt_embeds.shape
                            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                            prompt_embeds_list.append(prompt_embeds)

                        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
                        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

                        # Create Spatial-dimensional conditions.

                        original_size = (resolution, resolution)
                        target_size = (resolution, resolution)
                        crops_coords_top_left = (
                            crops_coords_top_left_h,
                            crops_coords_top_left_w,
                        )
                        add_time_ids = list(
                            original_size + crops_coords_top_left + target_size
                        )
                        add_time_ids = torch.tensor([add_time_ids])

                        add_time_ids = add_time_ids.to(
                            device, dtype=prompt_embeds.dtype
                        ).repeat(bs_embed, 1)

                        added_kw = {
                            "text_embeds": pooled_prompt_embeds,
                            "time_ids": add_time_ids,
                        }

                        # Sample noise that we'll add to the latents

                        generator = torch.Generator()
                        generator.manual_seed(step)

                        noise = torch.randn(vae_latent.size(), generator=generator)
                        noise = noise.to(vae_latent.device).to(vae_latent.dtype)

                        bsz = vae_latent.shape[0]

                        timesteps = torch.randint(
                            0,
                            noise_scheduler.config.num_train_timesteps,
                            (bsz,),
                            generator=generator,
                        )
                        timesteps = timesteps.to(vae_latent.device).long()

                        noisy_model_input = noise_scheduler.add_noise(
                            vae_latent, noise, timesteps
                        )

                        # Predict the noise residual
                        model_pred = unet(
                            noisy_model_input,
                            timesteps,
                            prompt_embeds,
                            added_cond_kwargs=added_kw,
                        )[0]

                        loss = (model_pred - noise).pow(2)
                        loss = loss.mean()

                        val_loss += loss.item()
                        tot_n += model_pred.shape[0]

                        # progress_bar.update(1)
                        progress_bar.set_description(
                            f"# PTI :step: {global_step}, epoch: {epoch} Validatiion Loss: {val_loss / tot_n}"
                        )

                    wandb.log({"val_loss": val_loss / tot_n})

            global_step += 1


if __name__ == "__main__":
    main()