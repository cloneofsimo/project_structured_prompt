import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer

# Mock CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.load_image()  # Dummy image for testing
        prompts, coords = self.extract_prompts_coords(item)
        caption = item['caption']

        # Tokenize
        tokenized_prompts = [self.tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]
        tokenized_caption = self.tokenizer.encode(caption, add_special_tokens=True)

        return {
            "image": image,
            "prompts": tokenized_prompts,
            "coord": coords,
            "caption": tokenized_caption
        }

    def load_image(self):
        # Create a dummy image for testing (3x224x224)
        return torch.rand(3, 224, 224)

    def extract_prompts_coords(self, item):
        prompts = []
        coords = []
        for exp in item['ref_exps']:
            prompt = item['caption'][exp[0]:exp[1]]
            coord = [exp[2], exp[3], exp[4], exp[5]]
            prompts.append(prompt)
            coords.append(coord)
        return prompts, coords

def custom_collate(batch):
    images = torch.stack([item['image'] for item in batch])
    prompts = [item['prompts'] for item in batch]
    coords = [item['coord'] for item in batch]
    captions = [item['caption'] for item in batch]
    return {
        "image": images,
        "prompts": prompts,
        "coord": coords,
        "caption": captions
    }

# Create a mock dataset
mock_data = [
    {
        'url': 'dummy_url', 
        'caption': 'a wire hanger with a paper cover', 
        'ref_exps': [[0, 33, 0.1, 0.1, 0.8, 0.8]],
        'noun_chunks': [[0, 13, 0.1, 0.1, 0.8, 0.8]]
    }
    # Add more mock items if needed
]

# Instantiate dataset and dataloader
dataset = CustomDataset(mock_data)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate)

# Iterate through the dataloader
for batch in dataloader:
    print("Batch:", batch)
    break  # Only print the first batch for testing
