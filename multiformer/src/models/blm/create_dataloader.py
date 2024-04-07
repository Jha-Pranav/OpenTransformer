from datasets import load_from_disk
from src.tokenize.tokenizer import Tokenizer
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler
import functools


# Define collate function to handle padding
def collate_fn(batch:int,block_size:int,tokenizer:Tokenizer):
    print(batch)
    batch = [en[:block_size] for en in batch]  # TODO : Write code for overflow
    x_batch = [torch.tensor(en[:-1]) for en in batch]  # Extract x (remove last token)
    y_batch = [torch.tensor(en[1:]) for en in batch]  # Extract y (remove first token)
    x_padded = pad_sequence(
        x_batch, batch_first=True, padding_value=tokenizer.eos_id()
    )  # Pad x sequences
    y_padded = pad_sequence(
        y_batch, batch_first=True, padding_value=tokenizer.eos_id()
    )  # Pad y sequences
    return x_padded, y_padded


# Create a sampler that samples data based on length
class SampleByLen(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        indices = sorted(range(len(self.data_source)), key=lambda i: len(self.data_source[i]))
        return iter(indices)
    

# Create DataLoader with collate function
def data_iter(batch:int=24,block_size:int=768,data=None,tokenizer=None):
    if not tokenizer:
        print('>> Loading Tokenizer')
        BASE_URL = "/home/pranav-pc/projects/OpenTransformer/multiformer/"
        TOKENIZER_CHECKPOINT = os.path.join(BASE_URL, "tokenizer_checkpoints")
        
        tokenizer = Tokenizer(TOKENIZER_CHECKPOINT)
    if not data:
        print('>> Loading Data')
        data = load_from_disk(BASE_URL + "data/interim/TinyStories.hf")
    return DataLoader(
    data['idx'], batch_size=batch, collate_fn=functools.partial(collate_fn, block_size=block_size,tokenizer=tokenizer), shuffle=False,pin_memory=True,num_workers=os.cpu_count()-2)

