from datasets import load_from_disk
from src.tokenize.tokenizer import Tokenizer
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler
import functools


# Define collate function to handle padding
def collate_fn(batch: int, padding_id: int):
    batch = pad_sequence(
        (torch.LongTensor(_["idx"]) for _ in batch),
        batch_first=True,
        padding_value=padding_id,
    )  # TODO : ShortTensor suffice our need but nn.Embedding don't support it. Using LOngTensor is a unnecessary waste of GPU memory
    x_batch = torch.stack([en[:-1] for en in batch])  # Extract x (remove last token)
    y_batch = torch.stack([en[1:] for en in batch])  # Extract y (remove first token)
    return x_batch, y_batch


# Create a sampler that samples data based on length
class SampleByLen(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        indices = sorted(
            range(len(self.data_source)), key=lambda i: len(self.data_source[i])
        )
        return iter(indices)


# Create DataLoader with collate function
def data_iter(batch: int = 32, data=None, tokenizer=None):
    if not tokenizer:
        print(">> Loading Tokenizer")
        BASE_URL = "/home/pranav-pc/projects/OpenTransformer/multiformer/"
        TOKENIZER_CHECKPOINT = os.path.join(BASE_URL, "tokenizer_checkpoints")
        tokenizer = Tokenizer(TOKENIZER_CHECKPOINT)
    if not data:
        print(">> Loading Data")
        data = load_from_disk(BASE_URL + "data/interim/TinyStories_65>tk>512.hf")
    padding_id = tokenizer.eos_id()  # Same a end of Seq
    # Create DataLoader with custom subset random sampler
    return DataLoader(
        data,
        batch_size=batch,
        collate_fn=functools.partial(collate_fn, padding_id=padding_id),
        shuffle=False,
        pin_memory=True,
        num_workers=20,
    )
