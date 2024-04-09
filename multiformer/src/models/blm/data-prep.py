from datasets import load_dataset, concatenate_datasets

BASE_URL = "/home/pranav-pc/projects/OpenTransformer/multiformer/"


def read_data():

    data_train = load_dataset(
        BASE_URL + "data/downloads/TinyStories",
        split="train",
    )
    data_validation = load_dataset(
        BASE_URL + "data/downloads/TinyStories",
        split="validation",
    )
    dataset = concatenate_datasets([data_train, data_validation])
    return dataset


dataset = read_data()

from src.tokenize.tokenizer import Tokenizer


def load_tokenizer():
    TOKENIZER_CHECKPOINT = BASE_URL + "tokenizer_checkpoints/"

    tokenizer = Tokenizer(TOKENIZER_CHECKPOINT)
    return tokenizer


tokenizer = load_tokenizer()
# ---------------
batch_size = int(1e5)
batched = True
num_proc = 25

min_seq_len = 65
max_seq_len = 1024
# --------------
dataset = dataset.map(
    lambda x: {"text": [en.strip() for en in x["text"]]},
    batch_size=batch_size,
    batched=batched,
    num_proc=num_proc,
)

dataset = dataset.map(
    lambda example: {"idx": tokenizer.encode(example["text"])},
    batch_size=batch_size,
    batched=batched,
    num_proc=num_proc,
    remove_columns=["text"],
)

dataset = dataset.map(
    lambda example: {"len": len(example["idx"])}
)
dataset = dataset.sort("len")
dataset = dataset.filter(
    lambda x: x["len"] > min_seq_len,
    batch_size=batch_size,
    batched=batched,
    num_proc=num_proc,
)  # remove sample with less than 66 tokens


def pack_dataset(dataset, batch_size, batched, num_proc):
    from src.data_wrangling.pack_overflow_seq import pack_seq

    dataset = dataset.map(
        lambda sample: pack_seq(sample, min_seq_len=65, max_seq_len=1024),
        batched=batched,
        batch_size=batch_size,
        num_proc=num_proc,
    )

    packed_data = dataset.remove_columns(["idx", "len", "overflow"]).rename_column(
        "packed", "idx"
    )
    overflow_data = dataset.remove_columns(["idx", "len", "packed"]).rename_column(
        "overflow", "idx"
    )

    packed_data = packed_data.filter(
        lambda x: x["tokens"], num_proc=num_proc, batched=batched, batch_size=batch_size
    )
    overflow_data = overflow_data.filter(
        lambda x: x["tokens"], num_proc=num_proc, batched=batched, batch_size=batch_size
    )

    dataset = concatenate_datasets([packed_data, overflow_data])
    return dataset


dataset = pack_dataset(dataset, batch_size, batched, num_proc)

dataset = dataset.map(
    lambda example: {"len": len(example["idx"])}
)
dataset = dataset.sort("len")

dataset = pack_dataset(dataset, batch_size, batched, num_proc)
dataset = dataset.remove_columns("len")

dataset.save_to_disk(
    BASE_URL + f"data/interim/TinyStories_{min_seq_len}>tk>{min_seq_len}.hf"
)
