import lightning.pytorch as pl
import torch
import torch.nn as nn
from bs4 import BeautifulSoup
from omegaconf import OmegaConf
from src.cells.utils.compile_utils import torch_compile
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("medium")
torch._dynamo.config.suppress_errors = True
# Silence all warnings
import warnings

import torch.nn.functional as F
from lightning.pytorch.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateFinder,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

warnings.filterwarnings("ignore")


class IMDBDataLoader(pl.LightningDataModule):
    def __init__(self, dataset_path, tokenizer_path, batch_size, num_workers, max_len):
        super().__init__()
        self.dataset_path = dataset_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len

        self.tokenizer = self._load_tokenizer(tokenizer_path)

    def prepare_data(self):
        from datasets import load_dataset

        self.ds = load_dataset(self.dataset_path)
        # self.ds = self.ds.map(lambda example : {'text':self._remove_html_tags(example['text'])},num_proc=self.num_workers,)
        self.label_map = {0: "neg", 1: "pos"}

    @staticmethod
    def _load_tokenizer(tokenizer_path):
        from src.tokenize.tokenizer import Tokenizer

        return Tokenizer(tokenizer_path)

    @staticmethod
    def _remove_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        # Get the text without HTML tags
        clean_text = soup.get_text()
        return clean_text

    def _collate_fn(self, batch):
        x, y = [self._remove_html_tags(en["text"]) for en in batch], [en["label"] for en in batch]
        x = [torch.tensor(tokens[: self.max_len]) for tokens in self.tokenizer.encode_as_ids(x)]
        x = pad_sequence(
            x,
            batch_first=True,
            padding_value=self.tokenizer.eos_id(),
        )
        y = torch.tensor(y)
        return x, y

    def setup(self, stage):
        self.train_data = self.ds["train"]
        self.val_data = self.ds["unsupervised"]
        self.test_data = self.ds["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )


import torchmetrics


class BLMClassifierModel(pl.LightningModule):
    def __init__(self, model, learning_rate=5e-5, num_classes=2, embedding_dim=768):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model

        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=2)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids):
        return self.model(input_ids)

    def _common_step(self, batch):
        x, label = batch
        hidden_state = self.forward(x)
        logits = self.classifier(hidden_state[:, -1, :])
        return logits

    def training_step(self, batch, batch_idx):
        x, targets = batch
        logits = self._common_step(batch)
        loss = F.cross_entropy(logits, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        logits = self._common_step(batch)

        loss = F.cross_entropy(logits, targets)
        preds = torch.argmax(logits, dim=1)
        val_acc = self.acc(preds, targets)
        val_f1score = self.f1_score(preds, targets)
        self.log_dict(
            {"val_loss": loss, "val_acc": val_acc, "val_f1score": val_f1score}, prog_bar=True
        )
        return loss

    # def test_step(self, batch, batch_idx):
    #     x, targets = batch
    #     logits = self._common_step(batch)

    #     loss = F.cross_entropy(logits, targets)
    #     preds = torch.argmax(logits, dim=1)
    #     test_acc = self.acc(preds, targets)
    #     test_f1score = self.f1_score(preds, targets)
    #     self.log_dict({"test_loss": loss,"test_acc":test_acc,"test_f1score":test_f1score}, prog_bar=False)
    #     return loss

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": 1e-2},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        lr_scheduler_init = {"T_max": 1e04, "eta_min": 1e-04}
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.learning_rate, betas=(0.9, 0.95), fused=False
        )
        scheduler = {
            "scheduler": CosineAnnealingLR(optimizer, **lr_scheduler_init),
            "interval": "step",
            "frequency": 10,
        }
        return [optimizer], [scheduler]


def main(args):
    ds = IMDBDataLoader(
        args.files.data_path,
        args.files.tokenizer_path,
        args.trainer_params.batch_size,
        args.trainer_params.num_workers,
        1024,
    )

    from src.models.blm.pl_training import Transformer

    MODEL_CHECKPOINT = args.paths.base_model_checkpoint
    base_model = Transformer.load_from_checkpoint(MODEL_CHECKPOINT)

    model = BLMClassifierModel(base_model)
    model.load_state_dict(torch.load(args.paths.resume_from_checkpoint)["state_dict"])
    model = torch_compile(model, dynamic=True, TORCH_COMPILE_BACKEND="inductor")
    accumulator = GradientAccumulationScheduler(
        scheduling=args.trainer_params.gradient_accumulation_scheduler
    )

    logger = TensorBoardLogger(save_dir="./lightning-log-ft-imdb/", name="IMDB", version=0.2)

    if args.trainer_params.wandb_enabled:
        import wandb

        print("W&B")
        wandb.login()
        logger = WandbLogger(**args.trainer_params.wandb)

    checkpoint_callback = ModelCheckpoint(**args.trainer_params.checkpoint)
    early_stop = EarlyStopping(**args.trainer_params.earlystopping)
    stochastic_weight_avg = StochasticWeightAveraging(swa_lrs=1e-6)

    trainer = pl.Trainer(
        logger=logger,
        **args.trainer_params.trainer,
        callbacks=[
            early_stop,
            checkpoint_callback,
            accumulator,
            LearningRateMonitor(logging_interval="step"),
            stochastic_weight_avg,
            # DeviceStatsMonitor()
        ],
    )
    model.train()
    if args.trainer_params.resume_training:
        trainer.fit(model, ds, ckpt_path=args.paths.resume_from_checkpoint)
    else:
        trainer.fit(model, ds)


if __name__ == "__main__":
    config_path = "./src/models/blm/conf/finetune-imdb-classifier.yaml"
    args = OmegaConf.load(config_path)
    main(args)
