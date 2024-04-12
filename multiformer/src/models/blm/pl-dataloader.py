import torch

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import functools

from datasets import load_from_disk

pl.seed_everything(123, workers=True)


class TinyStoriesDataloader(pl.LightningDataModule):
    def __init__(
        self, data_path_train, data_path_val, tokenizer_path, batch_size, num_workers
    ):
        super().__init__()
        self.data_path_train = data_path_train
        self.data_path_val = data_path_val

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tokenizer = self._load_tokenizer(tokenizer_path)

    def prepare_data(self):
        pass

    def _load_tokenizer(self, tokenizer_path):
        from src.tokenize.tokenizer import Tokenizer

        return Tokenizer(tokenizer_path)

    def _collate_fn(self, batch: int, padding_id: int):
        batch = pad_sequence(
            (torch.LongTensor(_["idx"]) for _ in batch),
            batch_first=True,
            padding_value=padding_id,
        )  # TODO : ShortTensor suffice our need but nn.Embedding don't support it. Using LOngTensor is a unnecessary waste of GPU memory
        x_batch = torch.stack(
            [en[:-1] for en in batch]
        )  # Extract x (remove last token)
        y_batch = torch.stack(
            [en[1:] for en in batch]
        )  # Extract y (remove first token)
        return x_batch, y_batch

    def setup(self, stage):

        self.train_data = load_from_disk(self.data_path_train)
        self.val_data = load_from_disk(self.data_path_val)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=functools.partial(
                self._collate_fn, padding_id=self.tokenizer.eos_id()
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=functools.partial(
                self._collate_fn, padding_id=self.tokenizer.eos_id()
            ),
        )
