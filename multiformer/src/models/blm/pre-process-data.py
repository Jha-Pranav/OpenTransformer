from datasets import load_dataset, concatenate_datasets
data_train = load_dataset('/home/pranav-pc/projects/OpenTransformer/multiformer/data/downloads/TinyStories',split='train')
data_validation = load_dataset('/home/pranav-pc/projects/OpenTransformer/multiformer/data/downloads/TinyStories',split='validation')
dataset = concatenate_datasets([data_train,data_validation])

from src.tokenize.tokenizer import Tokenizer

TOKENIZER_CHECKPOINT = (
    "/home/pranav-pc/projects/OpenTransformer/multiformer/tokenizer_checkpoints/"
)

tokenizer = Tokenizer(TOKENIZER_CHECKPOINT)

data = dataset.map(
    lambda example: {"idx": tokenizer.encode(example["text"])},
    batch_size=512,
    batched=True,
    remove_columns=["text"],
)
print(data)
data.save_to_disk(
    "/home/pranav-pc/projects/OpenTransformer/multiformer/data/interim/TinyStories.hf"
)
