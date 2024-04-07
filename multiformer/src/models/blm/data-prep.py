from datasets import load_dataset, concatenate_datasets
BASE_URL = "/home/pranav-pc/projects/OpenTransformer/multiformer/"


data_train = load_dataset(
    BASE_URL + "data/downloads/TinyStories",
    split="train",
)
data_validation = load_dataset(
    BASE_URL + "data/downloads/TinyStories",
    split="validation",
)
dataset = concatenate_datasets([data_train, data_validation])

from src.tokenize.tokenizer import Tokenizer

TOKENIZER_CHECKPOINT = (
   BASE_URL + "tokenizer_checkpoints/"
)

tokenizer = Tokenizer(TOKENIZER_CHECKPOINT)

dataset = dataset.map(
    lambda example: {"idx": tokenizer.encode(example["text"])},
    batch_size=512,
    batched=True,
    remove_columns=["text"],
)

dataset = dataset.map(lambda example: {"count": len(example["idx"])})
dataset = dataset.sort("count")
dataset = dataset.filter(
    lambda x: x["count"] > 10
)  # remove sample with less than 11 tokens
dataset = dataset.remove_columns(["count"])

dataset.save_to_disk(
   BASE_URL +  "data/interim/TinyStories.hf"
)
