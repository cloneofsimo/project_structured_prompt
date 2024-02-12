from huggingface_hub import hf_hub_download, login
from huggingface_hub import snapshot_download


#hf_hub_download(repo_id="JourneyDB/JourneyDB", filename = "data/train/train_anno_realease_repath.jsonl.tgz", local_dir = "./JourneyDB", repo_type="dataset")
#hf_hub_download(repo_id="JourneyDB/JourneyDB", filename = "data/train/imgs/000.tgz", local_dir = "./JourneyDB", repo_type="dataset")

import multiprocessing as mp
from huggingface_hub import hf_hub_download

# Define the function to download a file
def download_file(args):
    repo_id, filename, local_dir, repo_type = args
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, repo_type=repo_type, cache_dir = "./JourneyDB")

# Create a list of arguments for each download
download_args = [
    # ("JourneyDB/JourneyDB", "data/train/imgs/001.tgz", "./JourneyDB", "dataset"),
    # ("JourneyDB/JourneyDB", "data/train/imgs/002.tgz", "./JourneyDB", "dataset"),
    # ("JourneyDB/JourneyDB", "data/train/imgs/003.tgz", "./JourneyDB", "dataset"),
    # ("JourneyDB/JourneyDB", "data/train/imgs/004.tgz", "./JourneyDB", "dataset"),
    ("JourneyDB/JourneyDB", "data/train/imgs/005.tgz", "./JourneyDB", "dataset"),
    ("JourneyDB/JourneyDB", "data/train/imgs/006.tgz", "./JourneyDB", "dataset"),
    ("JourneyDB/JourneyDB", "data/train/imgs/007.tgz", "./JourneyDB", "dataset"),
    ("JourneyDB/JourneyDB", "data/train/imgs/008.tgz", "./JourneyDB", "dataset"),
    ("JourneyDB/JourneyDB", "data/train/imgs/009.tgz", "./JourneyDB", "dataset"),
    ("JourneyDB/JourneyDB", "data/train/imgs/010.tgz", "./JourneyDB", "dataset"),
]

# Create a pool of workers and start the downloads
with mp.Pool() as pool:
    pool.map(download_file, download_args)