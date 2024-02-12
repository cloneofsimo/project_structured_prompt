import torch
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Function to load the state dicts
def load_checkpoint(filepath):
    return load_file(filepath, device='cpu')

# Filepaths of your checkpoints
checkpoints = [
    "checkpoint_lr_high/unet/checkpoint-100.unet.safetensors",
    "checkpoint_lr_high/unet/checkpoint-200.unet.safetensors",
    "checkpoint_lr_high/unet/checkpoint-300.unet.safetensors",
    "checkpoint_lr_high/unet/checkpoint-400.unet.safetensors",
    "checkpoint_lr_high/unet/checkpoint-500.unet.safetensors",
    "checkpoint_lr_high/unet/checkpoint-600.unet.safetensors",
    "checkpoint_lr_high/unet/checkpoint-700.unet.safetensors",
    "checkpoint_lr_high/unet/checkpoint-800.unet.safetensors",
    "checkpoint_lr_high/unet/checkpoint-900.unet.safetensors",
    "checkpoint_lr_high/unet/checkpoint-1000.unet.safetensors",
]
# Function to extract the last three elements of the module name
def get_group_name(module_name):
    parts = module_name.split('.')
    if len(parts) >= 3:
        return '.'.join(parts[-3:])
    else:
        return module_name
# Load the 0th checkpoint
initial_state_dict = load_checkpoint(checkpoints[0])
print(initial_state_dict)

# Compute L2 distances
l2_distances = defaultdict(lambda: defaultdict(list))
for cp in checkpoints[1:]:
    current_state_dict = load_checkpoint(cp)
    for key in initial_state_dict.keys():
        group_name = get_group_name(key)
        l2_dist = torch.norm(initial_state_dict[key] - current_state_dict[key], 2)
        l2_distances[group_name][key].append(l2_dist.item())


checkpoints_x = range(1, len(checkpoints))
for group, distances_dict in l2_distances.items():
    # Average distance per group at each checkpoint
    avg_distances = np.mean([d for d in distances_dict.values()], axis=0)
    plt.plot(checkpoints_x, avg_distances, label=group)

plt.xlabel('Checkpoint')
plt.ylabel('Average L2 Distance')
plt.title('Average L2 Distance of Model Parameters by Group Over Time')
plt.legend()
plt.show()
# save figure
plt.savefig('l2_distance_high.png')