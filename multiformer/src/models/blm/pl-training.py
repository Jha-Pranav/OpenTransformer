import torch
from torch import nn
import torch.nn.functional as F
import math

from src.models.blm.config import ModelArgs
from src.models.blm.block import Block

from src.cells.normalization import RMSLayerNorm
from typing import Optional
from src.cells.position import RotaryEmbedding
from src.cells.optim_func import config_optimizer

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import functools

from datasets import load_from_disk, load_dataset
from tqdm.notebook import tqdm

import torch._dynamo
torch._dynamo.config.suppress_errors = True

torch.manual_seed(123)
torch.cuda.manual_seed(123)

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


class Transformer(pl.LightningModule):
    def __init__(self, args: ModelArgs):
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

        self.layers = nn.ModuleList([Block(args) for lid in range(args.num_layers)])

        self.norm = RMSLayerNorm(args.emebdding_dim, eps=args.rms_norm_eps)
        self.output = nn.Linear(args.emebdding_dim, args.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embd.weight = (
            self.output.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * args.num_layers)
                )
        self.lr = 1e-4
        
    def __repr__(self):
        return f"{self.get_num_params()} Million Params Model"

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
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

    def forward(
        self, tokens: torch.Tensor
    ) -> torch.Tensor:
        x = self.dropout(self.tok_embd(tokens))
        for layer in self.layers:
            x = layer(
                x, self.rope_q, self.rope_k
            )  ## How about we add residual connection here also ?
        x = self.norm(x)
        return x

    def _common_step(self,batch,batch_index):
        x, targets = batch
        logits = self.output(self.forward(x))
        loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return loss
        
    def training_step(self,batch,batch_idx):
        x,y = batch
        
        loss = self._common_step(batch,batch_idx)
        self.log_dict({'train_loss':loss},prog_bar=True,on_step=False,on_epoch=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y = batch
        
        loss = self._common_step(batch,batch_idx)
        self.log_dict({'val_loss':loss},prog_bar=True,on_step=False,on_epoch=True)
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
        return torch.optim.AdamW(optim_groups,lr=self.lr,betas=(0.9, 0.95),fused=False)

        
    def predict_step(self,batch,batch_idx,max_new_tokens=30,temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # trim the token to the max_len 
            if batch.shape[1] > self.max_seq_len:
                batch = batch[:,-self.max_seq_len:]
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
        return batch
    
if __name__=="__main__":
    BASE_URL = "/home/pranav-pc/projects/OpenTransformer/multiformer"
    data_path_train = BASE_URL + "/data/interim/TinyStories_train_65>tk>512.hf"
    data_path_val = BASE_URL + "/data/interim/TinyStories_val_65>tk>512.hf"
    tokenizer_path = BASE_URL + "/tokenizer_checkpoints"

    batch_size = 16
    num_workers = 26
    ds = TinyStoriesDataloader(
        data_path_train, data_path_val, tokenizer_path, batch_size, num_workers
    )

    conf = {
        "vocab_size": 32000,
        "emebdding_dim": 768,
        "max_seq_len": 512,
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
        "num_layers": 4,
        "device": ("cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backend.mps.is_available() else "cpu"),
        "padding_idx": ds.tokenizer.eos_id(),
    }

    config = ModelArgs(**conf)
    model = Transformer(config)
    model = torch.compile(model,dynamic=True)

    from lightning.pytorch.callbacks import GradientAccumulationScheduler, StochasticWeightAveraging, ModelCheckpoint,EarlyStopping
    accumulator = GradientAccumulationScheduler(scheduling={0: 6, 4: 4, 8: 3, 20:1})

    logger = pl.loggers.TensorBoardLogger(save_dir='./blm-log/', name='blm', version=0.1)
    # profiler = pl.profilers.PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./blm-log/'),
    #     schedule=torch.profiler.schedule(skip_first=10, wait=10, warmup=1, active=2)
    # )
    # saves top-K checkpoints based on "train_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        dirpath="/home/pranav-pc/projects/OpenTransformer/multiformer/model_checkpoints/blm/",
        filename="baby-llm-{epoch:02d}-{train_loss:.3f}",
        save_last= True,
        every_n_train_steps= int(1e4),
        save_on_train_epoch_end= True
    )
    early_stop = EarlyStopping('train_loss',patience=10,verbose=True)
    stochastic_weight_avg = StochasticWeightAveraging(swa_lrs=1e-2)

    trainer = pl.Trainer(
        logger=logger,
        min_epochs=1,
        max_epochs=100,
        precision='bf16-mixed',
        enable_model_summary=True,
        # profiler=profiler,
        callbacks=[early_stop,checkpoint_callback,accumulator,stochastic_weight_avg],
        default_root_dir="/home/pranav-pc/projects/OpenTransformer/multiformer/model_checkpoints/blm/",
        enable_checkpointing  = True,
        # fast_dev_run=True,
        log_every_n_steps=int(1e2),
        enable_progress_bar = True,
        gradient_clip_val=1.0)
    torch.set_float32_matmul_precision('medium')
    model.train()
    trainer.fit(model, ds,ckpt_path="/home/pranav-pc/projects/OpenTransformer/multiformer/model_checkpoints/blm/last.ckpt")
