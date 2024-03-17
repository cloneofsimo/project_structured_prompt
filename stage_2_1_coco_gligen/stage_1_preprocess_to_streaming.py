import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from diffusers.models import AutoencoderKL
from streaming import MDSWriter

import logging
import time
import numpy as np
from typing import Any

# Initialize logging
logging.basicConfig(level=logging.INFO)


from streaming.base.format.mds.encodings import Encoding, _encodings


class np32(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.float32)


_encodings["np32"] = np32


def crop_to_center(image, new_size=768):
    width, height = image.size

    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def prepare_image(pil_image, w=512, h=512):
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr)
    return image


class ImageDataset(Dataset):
    def __init__(self, root_dir, is_test=False):
        self.root_dir = root_dir
        self.image_paths = []
        self.captions = []
        if root_dir.endswith('.json'):
            # Read data from json file
            with open(root_dir, "r") as f:
                data = json.load(f)
                for d in data:
                    self.image_paths.append(d['img_path'])
                    self.captions.append(d['prompt'])

        if is_test:
            self.image_paths = self.image_paths[:200]
            self.captions = self.captions[:200]

        print("Number of images: ", len(self.image_paths))
        assert len(self.image_paths) == len(
            self.captions
        ), "Number of images and captions should match"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_file = self.image_paths[idx]
        caption = self.captions[idx]

        image = Image.open(image_file).convert("RGB")
        image = prepare_image(image, w = 768, h = 768)
       

        return image, caption


from tqdm import tqdm
from torch.utils.data import DataLoader

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
torch.set_float32_matmul_precision("high")


@torch.no_grad()
def convert_to_mds(
    root_dir, out_root, device, batch_size=8, num_workers=4, is_test=False
):
    logging.info(f"Processing on {device}")

    # Load the VAE model
    vae_model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    vae_model = vae_model.to(device).eval()
    vae_model.to(memory_format=torch.channels_last)
    # vae_model.encode = torch.compile(vae_model.encode, mode="reduce-overhead", fullgraph=False)

    # Create the dataset and dataloader
    dataset = ImageDataset(root_dir, is_test=is_test)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    sub_data_root = os.path.join(out_root, "data")
    columns = {"vae_output": "np32", "caption_output": "str"}

    if os.path.exists(sub_data_root):
        # Remove all files in the directory
        for file in os.listdir(sub_data_root):
            os.remove(os.path.join(sub_data_root, file))
    os.makedirs(sub_data_root, exist_ok=True)

    with MDSWriter(out=sub_data_root, columns=columns) as out:
        inference_latencies = []

        for batch in tqdm(dataloader):
            start_time = time.time()

            processed_images, captions = batch
            processed_images = processed_images.to(device)
            vae_outputs = vae_model.encode(processed_images).latent_dist.sample()

            # Iterate through the batch
            for i in range(len(captions)):
                sample = {
                    "vae_output": vae_outputs[i].cpu().numpy(),
                    "caption_output": captions[i],
                }
                out.write(sample)

            inference_latencies.append(time.time() - start_time)

        logging.info(
            f"Average Inference Latency on {device}: {np.mean(inference_latencies)} seconds"
        )


def main(root_dir, out_root, batch_size=32, num_workers=8, is_test=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    convert_to_mds(root_dir, out_root, device, batch_size, num_workers, is_test=is_test)
    logging.info("Finished processing images.")


if __name__ == "__main__":
    # Example usage
    i = 70
    root_dir = f"/root/bigdisk/project_structured_prompt/stage_2_1_coco_gligen/train.json"
    out_root = f"./coco_mds_train"
    # Set your output directory
    main(root_dir, out_root, is_test=False)

    # for i in range(70):
    #     # root_dir = './grit_files/00047'  # Set your dataset directory
    #     root_dir = f"/root/bigdisk/project_structured_prompt/stage_2_gligen_train/grit_files/{str(i).zfill(5)}"
    #     out_root = f"./grit_mds_train/{str(i).zfill(5)}"
    #     # Set your output directory
    #     main(root_dir, out_root)
