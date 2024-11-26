import os
from pathlib import Path
from typing import Optional, Union

from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self, checkpoint_dir: Path) -> None:
        checkpoint_path = os.path.join(checkpoint_dir, "tokenizer.model")
        assert os.path.exists(
            checkpoint_path
        ), f"The tokenizer.model file doesn't exist this {checkpoint_dir}"

        self.tokenizer = SentencePieceProcessor(checkpoint_path, add_bos=True, add_eos=True)

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.__init__(*args, **kwargs)
        return instance.tokenizer

    def __repr__(self):
        return self.tokenizer

    def stoid(self, word: str) -> int:
        """word to id"""
        return self.tokenizer.piece_to_id(word)

    def encode(self, docs: Union[list, str]):
        """
        docs : single str or list of str
        output ->
            list of tokens
        Note :1. The tokens are not padded so different entry can have different len
              2. Max length has also not been taken into consideration
        """
        # TODO  : Min/Max filtering and overflowing_tokens implementation
        return self.tokenizer.encode(docs)

    def decode(self, tokens: list) -> Union[list, str]:
        """
        tokens : list or (list of list) of tokens

        """
        return self.tokenizer.decode(tokens)


if __name__ == "__main__":
    import torch

    tokenizer = Tokenizer(
        "/home/pranav-pc/projects/OpenTransformer/multiformer/tokenizer_checkpoints/"
    )
    # Load dataset from Hugging Face datasets library
    # from datasets import load_dataset

    # dataset = load_dataset("imdb")

    # raw_text = dataset['test']['text']
    raw_text = "Let's have party tonight."
    tokenizer.decode(tokenizer.encode(raw_text))
