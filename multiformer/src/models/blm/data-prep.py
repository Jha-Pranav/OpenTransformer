from datasets import load_dataset, concatenate_datasets
from src.tokenize.tokenizer import Tokenizer


def read_data():

    data_train = load_dataset(
        BASE_URL + "/data/downloads/TinyStories",
        split="train",
    )
    data_validation = load_dataset(
        BASE_URL + "/data/downloads/TinyStories",
        split="validation",
    )
    return data_train, data_validation


def load_tokenizer():
    TOKENIZER_CHECKPOINT = BASE_URL + "/tokenizer_checkpoints/"

    tokenizer = Tokenizer(TOKENIZER_CHECKPOINT)
    return tokenizer


def text2tokens(
    dataset,
    tokenizer,
    batch_size: int,
    batched: int,
    num_proc: int,
    text_col: str = "text",
):
    dataset = dataset.map(
        lambda x: {"text": [en.strip() for en in x[text_col]]},
        batch_size=batch_size,
        batched=batched,
        num_proc=num_proc,
    )

    dataset = dataset.map(
        lambda example: {"idx": tokenizer.encode(example[text_col])},
        batch_size=batch_size,
        batched=batched,
        num_proc=1, # TODO: DEBUGG - Tokenizer don't add bos and eos in case of num_proc > 1. This is very weird.
        remove_columns=[text_col],
    )

    return dataset


def length(dataset):
    dataset = dataset.map(lambda example: {"len": len(example["idx"])})

    return dataset


def pack_dataset(
    dataset, min_seq_len, max_seq_len, tokenizer, batch_size, batched, num_proc
):
    from src.data_wrangling.pack_overflow_seq import pack_seq

    dataset = dataset.map(
        lambda sample: pack_seq(
            sample,
            min_seq_len,
            max_seq_len,
            delimiter_id=tokenizer.piece_to_id("."),
            eos_id=tokenizer.eos_id(),
            para_separator=tokenizer.piece_to_id("<0x0A>"),
            sos_id=tokenizer.bos_id(),
        ),
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
        lambda x: x["idx"], num_proc=num_proc, batched=batched, batch_size=batch_size
    )
    overflow_data = overflow_data.filter(
        lambda x: x["idx"], num_proc=num_proc, batched=batched, batch_size=batch_size
    )

    dataset = concatenate_datasets([packed_data, overflow_data])
    return dataset


def _pre_process(dataset, args):
    if not args.use_cache:
        dataset.cleanup_cache_files()

    tokenizer = load_tokenizer()

    dataset = text2tokens(
        dataset, tokenizer, args.batch_size, args.batched, args.num_proc, args.text_col
    )

    dataset = length(dataset)
    dataset = dataset.sort("len")

    dataset = dataset.filter(lambda x: x["len"] > args.min_seq_len)

    # fig = px.histogram(dataset["len"])
    # fig.update_layout(title_text=f"Histogram(len)_datasz_{len(dataset['len'])} ")
    # fig.write_html(f"TinyStories_{args.min_seq_len}>tk>{args.max_seq_len}_raw.html")
    # fig.show()

    for _ in range(1, 5):
        batch_scale = _
        dataset = pack_dataset(
            dataset,
            args.min_seq_len,
            args.max_seq_len,
            tokenizer,
            len(dataset["len"]) // batch_scale,
            args.batched,
            args.num_proc,
        )
        dataset = length(dataset)
        dataset = dataset.sort("len")
        # fig = px.histogram(dataset["len"])
        # fig.update_layout(
        #     title_text=f"Histogram(len) {_}_datasz_{len(dataset['len'])} itr"
        # )
        # fig.write_html(
        #     f"TinyStories_{args.min_seq_len}>tk>{args.max_seq_len}_itr{_}.html"
        # )
        # fig.show()
    # TODO : Below filter should not be required
    dataset = dataset.filter(lambda x: x["len"] < args.max_seq_len)

    dataset = dataset.remove_columns("len")
    return dataset


if __name__ == "__main__":
    import argparse
    import plotly.express as px
    import os

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--base_url",
        default=os.getcwd(),
        help="Base URL",
    )
    parser.add_argument("--batch_size", type=int, default=int(1e5), help="Batch size")
    parser.add_argument("--batched", default=True, help="Enable batching")
    parser.add_argument(
        "--num_proc", type=int, default=28, help="Number of processes"
    )  
    parser.add_argument(
        "--min_seq_len", type=int, default=65, help="Minimum sequence length"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument("--text_col", default="text", help="Text column name")
    parser.add_argument("--use_cache", default=False, help="Use cache")

    args = parser.parse_args()

    BASE_URL = args.base_url

    data_train, data_validation = read_data()

    data_train, data_validation = _pre_process(data_train, args), _pre_process(
        data_validation, args
    )

    data_train.save_to_disk(
        BASE_URL
        + f"/data/interim/TinyStories_train_{args.min_seq_len}>tk>{args.max_seq_len}.hf"
    )
    data_validation.save_to_disk(
        BASE_URL
        + f"/data/interim/TinyStories_val_{args.min_seq_len}>tk>{args.max_seq_len}.hf"
    )
