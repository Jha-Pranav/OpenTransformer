import functools

import lightning.pytorch as pl
import torch
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

pl.seed_everything(123, workers=True)


class TinyStoriesDataloader(pl.LightningDataModule):
    def __init__(
        self,
        data_path_train,
        data_path_val,
        tokenizer_path,
        batch_size,
        num_workers,
        subset_ratio,
    ):
        super().__init__()
        self.data_path_train = data_path_train
        self.data_path_val = data_path_val

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subset_ratio = (
            subset_ratio  # Only load subset of data while performing sanity check
        )
        assert 0 < self.subset_ratio <= 1, "the subset ratio can only lies b/w (0-1]"
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
        )  # TODO : ShortTensor suffice our need but nn.Embedding don't support it. Using LongTensor is a unnecessary waste of GPU memory
        x_batch = torch.stack(
            [en[:-1] for en in batch]
        )  # Extract x (remove last token)
        y_batch = torch.stack(
            [en[1:] for en in batch]
        )  # Extract y (remove first token)
        return x_batch, y_batch

    def setup(self, stage):
        self.train_data = load_from_disk(self.data_path_train)
        if self.subset_ratio and (self.subset_ratio != 1):
            self.train_data = self.train_data.select(
                range(int(len(self.train_data) * self.subset_ratio))
            )
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
