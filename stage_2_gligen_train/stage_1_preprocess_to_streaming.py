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

def adjust_bounding_box_and_caption_complex(noun_chunks, caption, original_width, original_height, crop_width=712, crop_height=712):
    """
    Adjusts bounding boxes based on the original image size and modifies the caption to include positional info.
    
    Parameters:
    - noun_chunks: List of tuples with noun chunk information.
    - caption: Original caption string.
    - original_width: Width of the original image before cropping.
    - original_height: Height of the original image before cropping.
    - crop_width: Width of the cropped image.
    - crop_height: Height of the cropped image.
    
    Returns:
    - Modified caption with positional info added.
    """
    # Calculate crop offsets if the image was centered before cropping
    offset_x = (original_width - crop_width) / 2 if original_width > crop_width else 0
    offset_y = (original_height - crop_height) / 2 if original_height > crop_height else 0
    
    # Sort noun_chunks by start position in reverse to not mess up the indexes when modifying the string
    noun_chunks = sorted(noun_chunks, key=lambda x: x[0], reverse=True)
    
    for chunk in noun_chunks:
        # Un-normalize bounding box coordinates to the original image size
        x_min = (chunk[2] * original_width) - offset_x
        y_min = (chunk[3] * original_height) - offset_y
        x_max = (chunk[4] * original_width) - offset_x
        y_max = (chunk[5] * original_height) - offset_y
        
        # Ensure the coordinates are within the cropped image bounds
        x_min = max(0, min(x_min, crop_width))
        y_min = max(0, min(y_min, crop_height))
        x_max = max(0, min(x_max, crop_width))
        y_max = max(0, min(y_max, crop_height))
        
        # Insert the positional info at the noun it is referring to
        position_info = f"<|{x_min:.0f}|><|{y_min:.0f}|><|{x_max:.0f}|><|{y_max:.0f}|>"
        start, end = int(chunk[0]), int(chunk[1])
        caption = caption[:start] + position_info + caption[start:]
    
    return caption



from streaming.base.format.mds.encodings import Encoding, _encodings


class np32(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.float32)


_encodings["np32"] = np32

def crop_to_center(image,  new_size = 768):
    width, height = image.size
   
    left = (width - new_size)/2
    top = (height - new_size)/2
    right = (width + new_size)/2
    bottom = (height + new_size)/2
    return image.crop((left, top, right, bottom))

def prepare_image(pil_image, w=512, h=512):
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr)
    return image

class ImageDataset(Dataset):
    def __init__(self, root_dir, is_test = False):
        self.root_dir = root_dir
        self.image_paths = []
        self.metadatas = []

        # Read data from directories
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(subdir, file)
                    try:
                        with open(json_path, 'r') as f:
                            metadata = json.load(f)
                    except:
                        print(f"Error reading {json_path}")
                        continue

                    self.metadatas.append(metadata)
                    # modify the file to be jpg
                    file_jpg = json_path.replace('.json', '.jpg')

                    self.image_paths.append(file_jpg)

        # cut down the dataset for testing 
        if is_test:
            self.image_paths = self.image_paths[:100]
            self.metadatas = self.metadatas[:100]

        print("Number of images: ", len(self.image_paths))
        assert len(self.image_paths) == len(self.metadatas), "Number of images and captions should match"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_file = self.image_paths[idx]
        metadata = self.metadatas[idx]

        image = Image.open(image_file).convert('RGB')
        cropped_image = crop_to_center(image, new_size=768)

        processed_image = prepare_image(cropped_image, w = 768, h = 768)

        caption = metadata['caption']
        noun_chunks = metadata['noun_chunks']
        original_width, original_height = image.size
        caption = adjust_bounding_box_and_caption_complex(noun_chunks, caption, original_width, original_height, crop_width=712, crop_height=712)


        return processed_image, caption

from tqdm import tqdm
from torch.utils.data import DataLoader
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
torch.set_float32_matmul_precision('high')

@torch.no_grad()
def convert_to_mds(root_dir, out_root, device, batch_size=8, num_workers=4, is_test = False):
    logging.info(f"Processing on {device}")

    # Load the VAE model
    vae_model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    vae_model = vae_model.to(device).eval()
    vae_model.to(memory_format=torch.channels_last)
    #vae_model.encode = torch.compile(vae_model.encode, mode="reduce-overhead", fullgraph=False)



    # Create the dataset and dataloader
    dataset = ImageDataset(root_dir, is_test= is_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

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

        logging.info(f"Average Inference Latency on {device}: {np.mean(inference_latencies)} seconds")

def main(root_dir, out_root, batch_size=32, num_workers=8, is_test=False, device_name='cuda'):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Processing on {device}")
    convert_to_mds(root_dir, out_root, device, batch_size, num_workers, is_test=is_test)
    logging.info("Finished processing images.")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert images to MDS format.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for processing (cuda or cpu).')
    parser.add_argument('--file_index', type=int, default=0, help='File index to process.')
    parser.add_argument('--is_test', action='store_true', help='Run in test mode with reduced dataset.')

    args = parser.parse_args()

    root_dir = f'./grit_files/{str(args.file_index).zfill(5)}'
    out_root = f"./grit_mds_train/{str(args.file_index).zfill(5)}"

    main(root_dir, out_root, is_test=args.is_test, device_name=args.device)

    # i = 55
    # root_dir = f'/root/bigdisk/project_structured_prompt/stage_2_gligen_train/grit_files/{str(i).zfill(5)}'
    # out_root = f"./grit_mds_test"
    #                 # Set your output directory
    # main(root_dir, out_root, is_test = True)

    # for i in range(55):
    #     #root_dir = '/root/bigdisk/project_structured_prompt/stage_2_gligen_train/grit_files/00047'  # Set your dataset directory
    #     root_dir = f'./grit_files/{str(i).zfill(5)}'
    #     out_root = f"./grit_mds_train/{str(i).zfill(5)}"
    #                     # Set your output directory
    #     main(root_dir, out_root)
