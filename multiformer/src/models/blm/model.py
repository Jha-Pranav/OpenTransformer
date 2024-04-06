import torch
from torch import nn
import torch.nn.functional as F


from src.models.blm.config import ModelArgs
from src.models.blm.block import Block

class BabyLangModel(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()

        self.tok_embd = nn.Embedding(args.vocab_size,args.emebdding_dim)
        self.layers = [nn.ModuleList(Block(args)) for lid in range(args.num_layers)]


