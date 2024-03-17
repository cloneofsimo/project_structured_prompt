import torch
import torch.nn.functional as F
import lightning as L
import re
import fnmatch
import json
import math
import os
import shutil
from typing import List, Optional
from streaming import StreamingDataset
import wandb

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
    output_dir: str = "./checkpoints/patch_pool2",
    seed: Optional[int] = 42,
    resolution: int = 768,
    crops_coords_top_left_h: int = 0,
    crops_coords_top_left_w: int = 0,
    train_batch_size: int = 6,
    num_train_epochs: int = 600,
    max_train_steps: Optional[int] = None,
    valid_steps: int = 300,  # default to no checkpoints
    checkpoint_steps: int = 300,
    gradient_accumulation_steps: int = 32,  # todo
    unet_learning_rate: float = 2e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 500,
    lr_num_cycles: int = 1,
    lr_power: float = 1.0,
    dataloader_num_workers: int = 0,
    max_grad_norm: float = 1.0,  # todo with tests
    allow_tf32: bool = True,
    mixed_precision: Optional[str] = "fp32",
    device: str = "cuda:0",
    verbose: bool = True,
    remote_train_dir="./coco_mds_train/data",
    remote_val_dir="./coco_mds_test/data",
) -> None:

    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if not seed:
        seed = np.random.randint(0, 2**32 - 1)
    print("Using seed", seed)
    torch.manual_seed(seed)
    latent_resolution = resolution // 8
    inserting_list_tokens = [f"<|{idx}|>" for idx in range(1000)]

    if mixed_precision == "fp16":
        weight_dtype = torch.float16
        fabric_precision = "16-mixed"

    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        fabric_precision = "bf16-mixed"
    else:
        weight_dtype = torch.float32
        fabric_precision = "32"

    fabric = L.Fabric(accelerator="cuda", devices=1, precision=fabric_precision)
    fabric.launch()

    global_loss = 0

    if fabric.global_rank == 0:
        wandb.init(
            project="Vendor",
            name=output_dir.split("/")[-1],
            config={
                "pretrained_model_name_or_path": pretrained_model_name_or_path,
                "output_dir": output_dir,
                "seed": seed,
                "resolution": resolution,
                "train_batch_size": train_batch_size,
                "num_train_epochs": num_train_epochs,
                "max_train_steps": max_train_steps,
                "valid_steps": valid_steps,
                "checkpoint_steps": checkpoint_steps,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "unet_learning_rate": unet_learning_rate,
                "scale_lr": scale_lr,
                "lr_scheduler": lr_scheduler,
                "lr_warmup_steps": lr_warmup_steps,
                "lr_num_cycles": lr_num_cycles,
                "lr_power": lr_power,
                "dataloader_num_workers": dataloader_num_workers,
                "max_grad_norm": max_grad_norm,
                "allow_tf32": allow_tf32,
                "mixed_precision": mixed_precision,
                "device": device,
                "verbose": verbose,
                "remote_train_dir": remote_train_dir,
                "remote_val_dir": remote_val_dir,
            },
        )

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

    text_encoders = [text_encoder_one, text_encoder_two]

    # Define patterns to identify parameters of interest in the UNet model
    WHITELIST_PATTERNS = [
        "*.attn*.weight",
        "*.resnet*.weight",  # Added pattern to include ResNet layers
    ]
    BLACKLIST_PATTERNS = ["*time*"]

    handler = TokenEmbeddingsHandler(text_encoders, [tok1, tok2])

    handler.initialize_new_tokens(inserting_list_tokens)

    unet_param_to_optimize = []
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

    # Separate text encoder parameters into embedding and others
    text_encoder_embedding_params = []
    text_encoder_other_params = []
    for text_encoder in text_encoders:
        for name, param in text_encoder.named_parameters():
            param.requires_grad = True
            if "token_embedding" in name:
                text_encoder_embedding_params.append(param)
            else:
                text_encoder_other_params.append(param)

    # Optimizer creation with different learning rates for different parameter groups
    params_to_optimize = [
        {
            "params": unet_param_to_optimize,
            "lr": unet_learning_rate,
        },
        {
            "params": text_encoder_other_params,
            "lr": unet_learning_rate * 2,
        },
        {
            "params": text_encoder_embedding_params,
            "lr": 2e-4,
            "weight_decay": 1e-3,
        },
    ]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        weight_decay=1e-3,
    )

    train_dataset = StreamingDataset(
        local=local_train_dir,
        remote=remote_train_dir,
        split=None,
        shuffle=True,
        shuffle_algo="naive",
        num_canonical_nodes=1,
    )

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
    # text_encoder_one = fabric.to_device(text_encoder_one).to(weight_dtype)
    # text_encoder_two = fabric.to_device(text_encoder_two).to(weight_dtype)
    text_encoder_one = fabric.setup(text_encoder_one)
    text_encoder_two = fabric.setup(text_encoder_two)

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
        "epic <|98|><|269|><|303|><|545|>castle landscape",
        "<|109|><|1|><|607|><|712|>A man carries <|98|><|269|><|303|><|545|><|426|><|253|><|589|><|580|>chickens as authorities enforced total evacuation of residents living near Taal volcano in Agoncillo town, Batangas province, southern Philippines on Thursday Jan. 16, 2020. Taal volcano belched smaller plumes of ash Thursday but shuddered continuously with earthquakes and cracked roads in nearby towns, which were blockaded by police due to fears of a bigger eruption. (AP Photo/Aaron Favila",
        "<|547|><|247|><|45|><|112|> bottle <|0|><|304|><|283|><|416|> toilet <|397|><|281|><|563|><|383|> sink <|819|><|1|><|140|><|82|> person",
        "An <|98|><|269|><|303|><|300|>astronaut flying in space, 4k, high resolution.",
        "An <|400|><|269|><|303|><|300|>astronaut flying in space, 4k, high resolution.",
        "<|528|><|3|><|135|><|725|> bicycle <|0|><|3|><|625|><|754|> person <|99|><|275|><|420|><|449|> cat",
        "<|96|><|316|><|357|><|467|> giraffe <|51|><|22|><|618|><|745|> giraffe",
    ]

    def half_values(string):
        # catch <|int|> and replace with <|int/2|>
        catch = re.findall(r"<\|\d+\|>", string)
        for c in catch:
            val = int(c[2:-2])
            string = string.replace(c, f"<|{val//16}|>")

        return string

    process_stringlist = lambda x: [half_values(i) for i in x]

    valid_prompts = process_stringlist(valid_prompts)

    # if fabric.global_rank == 0:
    #     wandb.watch(unet.module, log = 'all', log_freq = 40)
    #     wandb.watch(text_encoder_one, log = 'all', log_freq = 40)
    #     wandb.watch(text_encoder_two, log = 'all', log_freq = 40)

    for epoch in range(first_epoch, num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            progress_bar.update(1)
            progress_bar.set_description(f"# PTI :step: {global_step}, epoch: {epoch}")

            sts = process_stringlist(batch["caption_output"])
            print(sts)
            vae_latent = (
                batch["vae_output"].reshape(-1, 4, latent_resolution, latent_resolution)
                * 0.13025
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

                print(tok)

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

            loss = (model_pred - noise).pow(2).mean()
            fabric.backward(loss)

            if (step + 1) % gradient_accumulation_steps == 0:

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                wandb.log({"train_loss": loss.item()})

            # every step, we reset the embeddings to the original embeddings.

            if global_step % valid_steps == 10:

                if fabric.global_rank == 0:
                    if global_step % checkpoint_steps == 10 and global_step > 0:
                        os.makedirs(f"{output_dir}/{global_step}", exist_ok=True)
                        unet_path = f"{output_dir}/{global_step}/unet.pth"
                        text_encoder_one_path = (
                            f"{output_dir}/{global_step}/text_encoder_one.pth"
                        )
                        text_encoder_two_path = (
                            f"{output_dir}/{global_step}/text_encoder_two.pth"
                        )

                        # save all.
                        save_file(unet.state_dict(), unet_path)
                        save_file(text_encoder_one.state_dict(), text_encoder_one_path)
                        save_file(text_encoder_two.state_dict(), text_encoder_two_path)

                    from diffusers import StableDiffusionXLPipeline

                    pipe = StableDiffusionXLPipeline.from_pretrained(
                        pretrained_model_name_or_path,
                        unet=unet.module,
                        text_encoder=text_encoder_one.module,
                        text_encoder_2=text_encoder_two.module,
                        tokenizer=tok1,
                        tokenizer_2=tok2,
                        torch_dtype=weight_dtype,
                        use_safetensors=True,
                    ).to(fabric.device)
                    generator = torch.Generator().manual_seed(0)

                    images = pipe(
                        valid_prompts,
                        guidance_scale=8.0,
                        generator = [torch.Generator(device="cuda").manual_seed(0) for i in range(len(valid_prompts))],
                        width=resolution,
                        height=resolution,
                    ).images
                    log_images = []
                    for img, promp in zip(images, valid_prompts):
                        _img = wandb.Image(img, caption=promp)
                        log_images.append(_img)

                    wandb.log({"images": log_images})

                    del pipe

                # Validation Loss
                with torch.no_grad():
                    val_loss = 0.0
                    tot_n = 0.0
                    for step, batch in enumerate(val_dataloader):
                        # sts = batch["caption_output"]
                        sts = process_stringlist(batch["caption_output"])

                        vae_latent = (
                            batch["vae_output"].reshape(
                                -1, 4, latent_resolution, latent_resolution
                            )
                            * 0.13025
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
