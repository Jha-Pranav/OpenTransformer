from src.tokenize.tokenizer import Tokenizer

TOKENIZER_CHECKPOINT = (
    "/home/pranav-pc/projects/OpenTransformer/multiformer/tokenizer_checkpoints/"
)

# Load dataset from Hugging Face datasets library
from datasets import load_dataset

dataset = load_dataset("imdb")

tokenizer = Tokenizer(TOKENIZER_CHECKPOINT)

data = dataset.map(
    lambda example: {"idx": tokenizer.encode(example["text"])},
    batch_size=512,
    batched=True,
    remove_columns=["text"],
)

data.save_to_disk("/home/pranav-pc/projects/OpenTransformer/multiformer/data/interim/imdb.hf")
