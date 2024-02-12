import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def visualize_batch(batch):
    image = batch["image"][0]  # Get the first image in the batch
    prompts = batch["prompts"][0]
    coords = batch["coord"][0]
    caption = batch["caption"][0]

    # Convert the tensor image to PIL for display
    image = image.permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    image = Image.fromarray((image * 255).astype('uint8'), 'RGB')

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Decode the caption
    decoded_caption = dataset.tokenizer.decode(caption)
    print("Caption:", decoded_caption)

    # Draw bounding boxes and prompts
    for i, prompt in enumerate(prompts):
        coord = coords[i]
        rect = patches.Rectangle((coord[0] * 224, coord[1] * 224), (coord[2] - coord[0]) * 224, (coord[3] - coord[1]) * 224, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        decoded_prompt = dataset.tokenizer.decode(prompt)
        print("Prompt:", decoded_prompt)
        ax.text(coord[0] * 224, coord[1] * 224, decoded_prompt, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.show()

# Example usage:
for batch in dataloader:
    visualize_batch(batch)
    break  # Only visualize the first batch for testing
