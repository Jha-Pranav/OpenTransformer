from importlib.metadata import version
import torch

print("TORCH VERSION :", version("torch"))
device = "cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print('GPU  : ', device.upper())

from datasets import load_dataset

# Load only 2% of the data 
dataset = load_dataset("/home/pranav-pc/projects/OpenTransformer/multiformer/data/wikipedia/",split='train[:1%]')

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",trust_remote_code=True)

tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(),52000)