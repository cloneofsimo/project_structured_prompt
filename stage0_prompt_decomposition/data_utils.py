import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import numpy as np
import os
import hashlib
from deepspeed.accelerator import get_accelerator
import os

from torch.utils.data import Subset
import re
import random

import pandas as pd


class SpecificDataset:

    def __init__(self, output_path, seed, local_rank, dataset_name):

        self.dataset_name = "./g.csv"
        self.dataset_name_clean = "hcxkin"

        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        self.raw_dataset = pd.read_csv(self.dataset_name)
        self.raw_dataset.reset_index()
        self.raw_dataset = self.raw_dataset.to_dict("records")

    def get_prompt_and_chosen(self, sample):
        chats = sample["chat"]
        return chats

    def get_train_data(self):
        return self.raw_dataset[:-300]

    def get_eval_data(self):
        return self.raw_dataset[-300:]


def get_raw_dataset(dataset_name, output_path, seed, local_rank):

    return SpecificDataset(output_path, seed, local_rank, dataset_name)


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(
    local_rank,
    output_path,
    dataset_name,
    seed,
    split_name,
    data_split,
    split_index,
    data_size,
    rebuild=False,
):
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    # reindex each time when using local jsonfile since it's more likely to get modified
    if rebuild or (not os.path.isfile(index_file_name)) or (dataset_name == "jsonfile"):
        splits = [float(s) for s in data_split.split(",")]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(
                splits_index[index] + int(round(split * float(data_size)))
            )
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i] : splits_index[split_i + 1]
            ]
            np.save(shuffle_idx_split_file_name, shuffle_idx_split, allow_pickle=True)
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


class PromptDataset(Dataset):

    def __init__(
        self, prompt_dataset, chosen_dataset, reject_dataset, pad_token_id, train_phase
    ) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):

        return {
            "input_ids": self.chosen_dataset[idx]["input_ids"],
            "attention_mask": self.chosen_dataset[idx]["attention_mask"],
            "labels": self.chosen_dataset[idx]["input_ids"],
        }


def create_dataset_split(
    current_dataset,
    raw_dataset,
    train_phase,
    tokenizer,
    end_of_conversation_token,
    max_seq_len,
):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []

    print(current_dataset)

    for i, tmp_data in enumerate(current_dataset):
        # tokenize the text
        chosen_sentence = raw_dataset.get_prompt_and_chosen(
            tmp_data
        )  # the accept response
        if chosen_sentence is not None:
            chosen_sentence += end_of_conversation_token
            chosen_token = tokenizer(
                chosen_sentence,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(0)
            chosen_token["attention_mask"] = chosen_token["attention_mask"].squeeze(0)
            chosen_dataset.append(chosen_token)
    print(
        f"Creating dataset {raw_dataset.dataset_name_clean} for {train_phase=} size={len(chosen_dataset)}"
    )

    return PromptDataset(
        prompt_dataset,
        chosen_dataset,
        reject_dataset,
        tokenizer.pad_token_id,
        train_phase,
    )


def create_dataset(
    local_rank,
    dataset_name,
    data_split,
    output_path,
    train_phase,
    seed,
    tokenizer,
    end_of_conversation_token,
    max_seq_len,
    rebuild,
):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    train_dataset = raw_dataset.get_train_data()
    train_index = get_raw_dataset_split_index(
        local_rank,
        output_path,
        raw_dataset.dataset_name_clean,
        seed,
        "train",
        data_split,
        train_phase - 1,
        len(train_dataset),
        rebuild,
    )
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(
        train_dataset,
        raw_dataset,
        train_phase,
        tokenizer,
        end_of_conversation_token,
        max_seq_len,
    )

    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(
        local_rank,
        output_path,
        raw_dataset.dataset_name_clean,
        seed,
        "eval",
        data_split,
        train_phase - 1,
        len(eval_dataset),
        rebuild,
    )
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(
        eval_dataset,
        raw_dataset,
        train_phase,
        tokenizer,
        end_of_conversation_token,
        max_seq_len,
    )
    return train_dataset, eval_dataset


def create_prompt_dataset(
    local_rank,
    data_path,
    data_split,
    output_path,
    train_phase,
    seed,
    tokenizer,
    max_seq_len,
    end_of_conversation_token="<|endoftext|>",
    sft_only_data_path=[],
    rebuild=False,
):
    """
    Creates the prompt dataset
    """
    print(f"Called Prompt Gen")
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    sft_cache_key = "_".join(sft_only_data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(
        fname.encode()
    ).hexdigest()  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).to(
        get_accelerator().current_device_name()
    )
    torch.distributed.all_reduce(buf_create_cache)

    if local_rank <= 0 and rebuild:

        train_dataset, eval_dataset = create_dataset(
            local_rank,
            data_path[0],
            data_split,
            output_path,
            train_phase,
            seed,
            tokenizer,
            end_of_conversation_token,
            max_seq_len,
            rebuild=rebuild,
        )

        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
    else:
        if local_rank <= 0:
            print("Not rebuilding!!")

    torch.distributed.barrier()
    return torch.load(train_fname), torch.load(eval_fname)
