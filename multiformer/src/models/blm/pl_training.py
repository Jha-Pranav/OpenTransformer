import math
import lightning.pytorch as pl
import torch
import torch._dynamo
import torch.nn.functional as F
import wandb
from huggingface_hub import PyTorchModelHubMixin
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from src.cells.normalization import RMSLayerNorm
from src.cells.position import RotaryEmbedding
from src.models.blm.block import Block
from src.models.blm.config import ModelArgs
from src.models.blm.pl_dataloader import TinyStoriesDataloader
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from omegaconf import OmegaConf

torch.set_float32_matmul_precision("medium")
torch._dynamo.config.suppress_errors = True
from lightning.pytorch.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateFinder,
    ModelCheckpoint,
    StochasticWeightAveraging,
)

torch.manual_seed(123)
torch.cuda.manual_seed(123)
pl.seed_everything(123, workers=True)

class Transformer(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(self, args: ModelArgs, is_causal=True, attn_mask=None,lr=5e-4,cosine_t_max=int(1e4)):
        super().__init__()
        self.save_hyperparameters()
        self.max_seq_len = args.max_seq_len
        self.tok_embd = nn.Embedding(
            args.vocab_size, args.embedding_dim, padding_idx=args.padding_idx
        )
        self.dropout = nn.Dropout(args.embedding_dropout)
        self.rope_q = RotaryEmbedding(
            args.embedding_dim // args.num_attention_heads,
            args.max_seq_len,
            device=args.device,
        )
        self.rope_k = RotaryEmbedding(
            args.embedding_dim // args.num_key_value_heads,
            args.max_seq_len,
            device=args.device,
        )

        # Freeze the parameters rope_q and rope_k
        self.rope_q.requires_grad_(False)
        self.rope_k.requires_grad_(False)

        self.layers = nn.ModuleList(
            [Block(args, is_causal=is_causal, attn_mask=attn_mask) for _ in range(args.num_layers)]
        )

        self.norm = RMSLayerNorm(args.embedding_dim, eps=args.rms_norm_eps)
        self.output = nn.Linear(args.embedding_dim, args.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embd.weight = self.output.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * args.num_layers))
        self.lr = lr 
        self.cosine_t_max = cosine_t_max

    def __repr__(self):
        return f"{self.get_num_params()} Million Params Model"

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embd.weight.numel()
        return n_params / 1e6  # In Million

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.tok_embd(tokens))
        for layer in self.layers:
            x = layer(
                x, self.rope_q, self.rope_k
            )  ## How about we add residual connection here also ?
        x = self.norm(x)
        return x

    def _common_step(self, batch, batch_index):
        x, targets = batch
        logits = self.output(self.forward(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        # if self.trainer.global_step == 0:
        #     wandb.define_metric("train_loss", summary="mean")
        self.log_dict(
            {"train_loss": loss, "lr": self.lr}, prog_bar=True, on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log_dict({"val_loss": loss}, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": 1e-2},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        lr_scheduler_init = {"T_max": 1e04, "eta_min": 1e-06}
        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, betas=(0.9, 0.95), fused=True)
        scheduler = {
            "scheduler": CosineAnnealingLR(optimizer, **lr_scheduler_init),
            "interval": "step",
            "frequency":10
        }
        return [optimizer], [scheduler]

    def predict_step(
        self,
        batch,
        batch_idx,
        max_new_tokens=30,
        temperature=1.0,
        top_k=None,
        conditional_break: list = None,
    ):
        for _ in range(max_new_tokens):
            # trim the token to the max_len
            if batch.shape[1] > self.max_seq_len:
                batch = batch[:, -self.max_seq_len :]
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
                    last_three_tokens, torch.LongTensor(conditional_break).to(batch.device)
                ):
                    break

        return batch




def main(args):

    ds = TinyStoriesDataloader(
        args.files.data_path_train,
        args.files.data_path_val,
        args.files.tokenizer_path,
        args.trainer_params.batch_size,
        args.trainer_params.num_workers,
        args.trainer_params.subset_ratio,

    )
    model_conf = args.model
    model_conf["padding_idx"] = ds.tokenizer.eos_id()
    config = ModelArgs(**model_conf)
    model = Transformer(config)
    model = torch.compile(model, dynamic=True)

    accumulator = GradientAccumulationScheduler(
        scheduling=args.trainer_params.gradient_accumulation_scheduler
    )

    logger = TensorBoardLogger(save_dir='./lightning-log/', name='TinnyStories', version=0.1)

    if args.trainer_params.wandb_enabled:
        print('W&B')
        wandb.login()
        logger = WandbLogger(**args.trainer_params.wandb)

    checkpoint_callback = ModelCheckpoint(**args.trainer_params.checkpoint)
    early_stop = EarlyStopping(**args.trainer_params.earlystopping)
    # stochastic_weight_avg = StochasticWeightAveraging(swa_lrs=1e-6)
    from lightning.pytorch.callbacks import LearningRateMonitor
    trainer = pl.Trainer(
        logger=logger,
        **args.trainer_params.trainer,
        callbacks=[
            # early_stop,
            checkpoint_callback,
            accumulator,
            LearningRateMonitor(logging_interval='step'),
            # stochastic_weight_avg,
        ],
    )

    model.train()
    trainer.fit(model, ds)


if __name__ == "__main__":
    config = OmegaConf.load("src/models/blm/conf/config.yaml")
    main(config)
