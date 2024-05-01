import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_float32_matmul_precision("medium")
torch._dynamo.config.suppress_errors = True
from lightning.pytorch.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateFinder,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from src.cells.utils.compile_utils import torch_compile
from src.models.blm.pl_dataloader import TinyStoriesDataloader


class FineTuneBLM(pl.LightningModule):
    def __init__(self, model, learning_rate=5e-5, dropout_rate=0.1):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model
        self.output = self.model.output
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens)

    def _common_step(self, batch, batch_index):
        x, targets = batch
        logits = self.output(self.forward(x))
        logits = self.dropout(logits)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log_dict({"train_loss": loss}, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log_dict({"val_loss": loss}, prog_bar=True, on_step=True, on_epoch=False)
        return loss

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

    @torch.inference_mode()
    def predict_step(
        self,
        batch,
        batch_idx,
        max_new_tokens=30,
        temperature=1.0,
        top_k=None,
        conditional_break: list = None,
    ):
        input_len = batch.shape[1]
        for _ in range(max_new_tokens):
            # trim the token to the max_len
            if batch.shape[1] > self.model.max_seq_len:
                batch = batch[:, -self.model.max_seq_len :]
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(
                self(batch)[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            batch = torch.cat((batch, idx_next), dim=1)
            if conditional_break:
                last_three_tokens = batch[-1][-len(conditional_break) :]
                if torch.equal(
                    last_three_tokens,
                    torch.LongTensor(conditional_break).to(batch.device),
                ):
                    break
        return batch[:, input_len:]


def main(args):
    ds = TinyStoriesDataloader(
        args.files.data_path_train,
        args.files.data_path_val,
        args.files.tokenizer_path,
        args.trainer_params.batch_size,
        args.trainer_params.num_workers,
        args.trainer_params.subset_ratio,
    )

    from src.models.blm.pl_training import Transformer

    MODEL_CHECKPOINT = args.paths.base_model_checkpoint
    base_model = Transformer.load_from_checkpoint(MODEL_CHECKPOINT)

    model = FineTuneBLM(base_model)
    model = torch_compile(model, dynamic=True, TORCH_COMPILE_BACKEND="inductor")
    accumulator = GradientAccumulationScheduler(
        scheduling=args.trainer_params.gradient_accumulation_scheduler
    )

    logger = TensorBoardLogger(save_dir="./log-blm-instruct/", name="blm-instruct", version=0.1)

    if args.trainer_params.wandb_enabled:
        import wandb

        print("W&B")
        wandb.login()
        logger = WandbLogger(**args.trainer_params.wandb)

    # from lightning.pytorch.profilers import PyTorchProfiler

    # from lightning.pytorch.callbacks import DeviceStatsMonitor

    checkpoint_callback = ModelCheckpoint(**args.trainer_params.checkpoint)
    early_stop = EarlyStopping(**args.trainer_params.earlystopping)
    stochastic_weight_avg = StochasticWeightAveraging(swa_lrs=1e-6)
    from lightning.pytorch.callbacks import LearningRateMonitor

    trainer = pl.Trainer(
        logger=logger,
        **args.trainer_params.trainer,
        callbacks=[
            # early_stop,
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
    config_path = "./src/models/blm/conf/finetune-tinnystories-instruct.yaml"
    args = OmegaConf.load(config_path)
    main(args)
