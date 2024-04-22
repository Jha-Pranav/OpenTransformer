import argparse
import os

import plotly.express as px
from datasets import concatenate_datasets, load_dataset, load_from_disk
from src.tokenize.tokenizer import Tokenizer


def read_data(dataset_name, cache_path, processed_data=False):
    if processed_data:
        ds = load_from_disk(cache_path)
        data_train, data_validation = ds["train"], ds["validation"]
    else:
        data_train = load_dataset(
            dataset_name,
            cache_dir=cache_path,
            split="train",
        )
        data_validation = load_dataset(
            dataset_name,
            cache_dir=cache_path,
            split="validation",
        )
    return data_train, data_validation


def load_tokenizer(BASE_URL):
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
        num_proc=1,  # TODO: DEBUG - Tokenizer don't add bos and eos in case of num_proc > 1. This is very weird.
        remove_columns=[text_col],
    )

    return dataset


def length(dataset):
    dataset = dataset.map(lambda example: {"len": len(example["idx"])})

    return dataset


def pack_dataset(dataset, min_seq_len, max_seq_len, tokenizer, batch_size, batched, num_proc):
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

    packed_data = dataset.remove_columns(["idx", "len", "overflow"]).rename_column("packed", "idx")
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
    tokenizer = load_tokenizer(args.base_url)

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
    print(args.pack_dataset)
    if args.pack_dataset:
        for _ in range(1, 3):
            batch_scale = _
            dataset = pack_dataset(
                dataset,
                args.min_seq_len,
                args.max_seq_len,
                tokenizer,
                len(dataset["len"]) // batch_scale * 10000,
                args.batched,
                args.num_proc,
            )
            dataset = length(dataset)
            dataset = dataset.sort("len")
            fig = px.histogram(dataset["len"])
            fig.update_layout(title_text=f"Histogram(len) {_}_datasz_{len(dataset['len'])} itr")
            # fig.write_html(
            #     f"TinyStories_{args.min_seq_len}>tk>{args.max_seq_len}_itr{_}.html"
            # )
            fig.show()

    dataset = dataset.filter(lambda x: x["len"] < args.max_seq_len)

    dataset = dataset.remove_columns("len")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--base_url",
        default=os.getcwd(),
        help="Base URL",
    )
    parser.add_argument(
        "--processed_data", type=bool, default=True, help="Pass processed data path"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="hf dataset name eg: skeskinen/TinyStories-Instruct-hf",
    )
    parser.add_argument(
        "--dataset_cache_dir", type=str, default="./data/downloads", help="cache dir"
    )
    parser.add_argument("--batch_size", type=int, default=int(1e5), help="Batch size")
    parser.add_argument("--batched", default=True, help="Enable batching")
    parser.add_argument("--num_proc", type=int, default=28, help="Number of processes")
    parser.add_argument("--min_seq_len", type=int, default=65, help="Minimum sequence length")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--text_col", default="text", help="Text column name")
    parser.add_argument("--use_cache", type=bool, default=False, help="Use cache")
    parser.add_argument("--pack_dataset", type=bool, default=False, help="Pack Dataset")

    args = parser.parse_args()

    BASE_URL = args.base_url
    if not args.dataset_name:
        raise Exception("Dataset name is required.")

    data_train, data_validation = read_data(
        args.dataset_name, args.dataset_cache_dir, args.processed_data
    )

    data_train, data_validation = _pre_process(data_train, args), _pre_process(
        data_validation, args
    )

    data_train.save_to_disk(
        BASE_URL
        + f"/data/interim/{args.dataset_name}_train_{args.min_seq_len}>tk>{args.max_seq_len}.hf"
    )
    data_validation.save_to_disk(
        BASE_URL
        + f"/data/interim/{args.dataset_name}_val_{args.min_seq_len}>tk>{args.max_seq_len}.hf"
    )

# Eg command : python3 src/models/blm/data_prep.py --processed_data=true --dataset_name="TinyStories-Instruct-hf" --dataset_cache_dir="/home/pranav-pc/projects/OpenTransformer/multiformer/data/processed/TinyStories-Instruct-hf" --pack_dataset=false
