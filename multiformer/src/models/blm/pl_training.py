import math
import os

import lightning.pytorch as pl
import torch
import torch._dynamo
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from src.cells.normalization import RMSLayerNorm
from src.cells.position import RotaryEmbedding
from src.models.blm.block import Block
from src.models.blm.config import ModelArgs
from src.models.blm.pl_dataloader import TinyStoriesDataloader
from torch import nn

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


class Transformer(pl.LightningModule):
    def __init__(self, args: ModelArgs, is_causal=True, attn_mask=None):
        super().__init__()
        self.save_hyperparameters()
        self.max_seq_len = args.max_seq_len
        self.tok_embd = nn.Embedding(
            args.vocab_size, args.emebdding_dim, padding_idx=args.padding_idx
        )
        self.dropout = nn.Dropout(args.embedding_dropout)
        self.rope_q = RotaryEmbedding(
            args.emebdding_dim // args.num_attention_heads,
            args.max_seq_len,
            device=args.device,
        )
        self.rope_k = RotaryEmbedding(
            args.emebdding_dim // args.num_key_value_heads,
            args.max_seq_len,
            device=args.device,
        )

        # Freeze the parameters rope_q and rope_k
        self.rope_q.requires_grad_(False)
        self.rope_k.requires_grad_(False)

        self.layers = nn.ModuleList(
            [Block(args, is_causal=is_causal, attn_mask=attn_mask) for _ in range(args.num_layers)]
        )

        self.norm = RMSLayerNorm(args.emebdding_dim, eps=args.rms_norm_eps)
        self.output = nn.Linear(args.emebdding_dim, args.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embd.weight = self.output.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * args.num_layers))
        self.lr = 1e-4

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
        if trainer.global_step == 0:
            wandb.define_metric("train_loss", summary="mean")
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
        return torch.optim.AdamW(optim_groups, lr=self.lr, betas=(0.9, 0.95), fused=False)

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


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


if __name__ == "__main__":
    BASE_URL = os.getcwd()
    data_path_train = BASE_URL + "/data/interim/TinyStories_train_65>tk>1024.hf"
    data_path_val = BASE_URL + "/data/interim/TinyStories_val_65>tk>1024.hf"
    tokenizer_path = BASE_URL + "/tokenizer_checkpoints"

    batch_size = 8
    num_workers = 26
    ds = TinyStoriesDataloader(
        data_path_train, data_path_val, tokenizer_path, batch_size, num_workers
    )

    conf = {
        "vocab_size": 32000,
        "emebdding_dim": 768,
        "max_seq_len": 1024,
        "embedding_dropout": 0.0,
        "rms_norm_eps": 1e-05,
        "rope_scaling": 1.0,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "use_cache": True,
        "use_sliding_window": True,
        "residual_dropout": 0.1,
        "mlp_dropout": 0.0,
        "mlp_hidden_size": int(1.3 * 768),
        "num_layers": 6,
        "device": (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backend.mps.is_available()
            else "cpu"
        ),
        "padding_idx": ds.tokenizer.eos_id(),
    }

    config = ModelArgs(**conf)
    model = Transformer(config)
    model = torch.compile(model, dynamic=True)

    accumulator = GradientAccumulationScheduler(scheduling={0: 4, 4: 3, 10: 2})

    import wandb

    wandb.login()
    wandb_logger = WandbLogger(
        name="blm-1024",
        save_dir="blm-1024/",
        version="v2",
        offline=True,
        project="tiny-stories-768",
    )
    # profiler = pl.profilers.PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./blm-log/'),
    #     schedule=torch.profiler.schedule(skip_first=10, wait=10, warmup=1, active=2)
    # )
    # saves top-K checkpoints based on "train_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="train_loss",
        mode="min",
        dirpath="/model_checkpoints/",
        filename="baby-llm-{epoch:02d}-{train_loss:.3f}",
        save_last=True,
        every_n_train_steps=int(1e4),
        save_on_train_epoch_end=True,
    )
    early_stop = EarlyStopping("train_loss", patience=10, verbose=True)
    stochastic_weight_avg = StochasticWeightAveraging(swa_lrs=1e-6)
    lr_finder = FineTuneLearningRateFinder(milestones=(0, 6))

    trainer = pl.Trainer(
        logger=wandb_logger,
        min_epochs=1,
        max_epochs=6,
        precision="bf16-mixed",
        enable_model_summary=True,
        # profiler=profiler,
        callbacks=[
            early_stop,
            checkpoint_callback,
            accumulator,
            stochastic_weight_avg,
            lr_finder,
        ],
        default_root_dir=".",
        enable_checkpointing=True,
        # fast_dev_run=True,
        log_every_n_steps=15,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )

    model.train()
    trainer.fit(model, ds)
