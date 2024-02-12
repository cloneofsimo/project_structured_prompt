import json
import pandas as pd
import os
# Path to your .jsonl file
file_path = '/root/bigdisk/project_structured_prompt/stage0_prompt_decomposition/scripts/JourneyDB/data/train/train_anno_realease_repath.jsonl'

# Initialize an empty list or DataFrame
data_list = []

from PIL import Image
# Open the file and iterate over each line
with open(file_path, 'r') as file:
    for line in file:
        # Parse the JSON data from each line
        data = json.loads(line)
        # img_path should start with ./000 ~ ./004
        img_path = data['img_path']
        start_prefix = img_path[:5]
        checklists = [f"./00{i}" for i in range(11)]
        if start_prefix not in checklists:
            continue


        # Process and append the data (for example, to a list)
        try:
            base_path = "/root/bigdisk/project_structured_prompt/stage0_prompt_decomposition/scripts/JourneyDB/data/train/imgs"
            Image.open(os.path.join(base_path, data['img_path']))
            data_list.append({
                'img_path': os.path.join(base_path, data['img_path']),
                'style': ', '.join(data['Task1'].get('Style', [])),
                'content': ', '.join(data['Task1'].get('Content', [])),
                'atmosphere': ', '.join(data['Task1'].get('Atmosphere', [])),
                'caption': data['Task2'].get('Caption', ''),
                'prompt': data['prompt']
            })
        except:
            print(data)

# Convert the list to a DataFrame
df = pd.DataFrame(data_list)
# dedup if it has same prompt
df = df.drop_duplicates(subset=['prompt'])

# print how many rows have not-none entries for each column. By none, I mean empty string.

for col in df.columns:
    print(col, df[df[col] != ''][col].count())

# fill empty string with ' '.
for col in df.columns:
    df[col] = df[col].apply(lambda x: ' ' if x == '' else x)
df.to_csv('journeydb_subsampled.csv', index=False)