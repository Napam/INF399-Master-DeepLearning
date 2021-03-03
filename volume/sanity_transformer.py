from typing import ForwardRef
import torch
from torch import nn  
from torch.nn.parameter import Parameter
from utils import debug, debugt, debugs
import numpy as np 
from torch.nn import functional as F

def scaled_dot_product_attention(q, k ,v):
    qk = q@k.T
    dk = len(k)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.wq: torch.Tensor = nn.Linear(d_model, d_model)
        self.wk: torch.Tensor = nn.Linear(d_model, d_model)
        self.wv: torch.Tensor = nn.Linear(d_model, d_model)

    def forward(self, X):
        '''
        X: (S,E)
            S: Source sequence length
            E: Embedding dim (d_model)

        dk should be (embed_dim // num_heads)**0.5
        '''
        q = self.wq(X)
        k = self.wk(X)
        v = self.wv(X)
        a = F.softmax(q @ k.T / len(q)**0.5, dim=-1)
        return a@v


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mha = MultiHeadAttention(d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )
        self.layernorm2 = nn.LayerNorm(d_model) 

    def forward(self, X):
        h = self.mha(X)
        h = F.dropout(X, p=0)
        h2 = self.layernorm1(h + X)
        self.ff(h)
        return h

if __name__ == "__main__":
    # X = torch.rand(4,8)
    X = torch.tensor([
        [ 0,  0,10, 0],
        [ 0, 10, 0, 0],
        [10, 10, 0, 0],
    ], dtype=torch.float32)

    model = TransformerEncoderLayer(d_model=4)
    model(X)
    # debug(X)