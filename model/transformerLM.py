import math
import copy
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch import Tensor
import torch.jit  # this is needed to avoid a circular import
from torch.distributed.tensor import DTensor

# import lightning as L

from utils.distributed_utils import *

@dataclass
class ModelArgs:
    dim: int = 4096
    n_heads: int = 32
    vocab_size: int = -1  
    num_encoder_layers: int = 2
    max_seq_length: int = 2048
    dropout: float = 0.1
    n_layers: int = 8
    
def is_dtensor(tensor):
    return isinstance(tensor, DTensor)


class MultiheadAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        
        self.linear_Q = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.linear_K = nn.Linear(self.kdim, self.embed_dim, bias=bias)
        self.linear_V = nn.Linear(self.vdim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)  


    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) :

        tgt_len, bsz, embed_dim_to_check = query.size()
        assert self.embed_dim == embed_dim_to_check
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = self.embed_dim // self.num_heads
        assert head_dim * self.num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        scaling = float(head_dim) ** -0.5
        
        q = self.linear_Q(query)
        k = self.linear_K(key)
        v = self.linear_V(value)

        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for `attn_mask` in `nn.MultiheadAttention` is deprecated. "
                    "Use bool tensor instead.",
                    stacklevel=3,
                )
                attn_mask = attn_mask.to(torch.bool)
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, (
                f"Only float and bool types are supported for attn_mask, not {attn_mask.dtype}"
            )

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * self.num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for `key_padding_mask` in `nn.MultiheadAttention` is deprecated. "
                "Use bool tensor instead.",
                stacklevel=3,
            )
            key_padding_mask = key_padding_mask.to(torch.bool)
        
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(
            attn_output_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]

        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(tgt_len, bsz, self.embed_dim)
        )

    
        attn_output = self.out_proj(attn_output)  # type: ignore[has-type]

        return attn_output.type_as(query), None


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))  # Add batch dimension
        

    def forward(self, x):
        return x + self.pe[:x.size(0)]

# ---- helper to build one encoder layer ----
class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_args, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(num_heads=model_args.n_heads, dropout=dropout, embed_dim=model_args.dim)
        self.linear1 = nn.Linear(model_args.dim, model_args.dim * 16)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(model_args.dim * 16, model_args.dim)
        self.norm1 = nn.LayerNorm(model_args.dim)
        self.norm2 = nn.LayerNorm(model_args.dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, device=None):

        src = self.norm1(
            src
            + self.dropout1(self.self_attn(src, src, src)[0])
        )

        src = self.norm2(
            src
            + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(src)))))
        )

        return src


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])

# ---- full TransformerLM with  layers ----
class TransformerLM(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.embed = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.pos_encoder = PositionalEncoding(model_args.dim)

        self.layers = _get_clones(TransformerEncoderLayer(model_args), model_args.num_encoder_layers)

        self.norm = nn.LayerNorm(model_args.dim)
        self.fc = nn.Linear(model_args.dim, model_args.vocab_size)

        self._reset_parameters()
 

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x):   
        if self.embed:
            x = x.t() 
            x = self.embed(x)
            x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x) if self.norm else x 
        x = self.fc(x) if self.fc else x 
        return x


# class LitTransformerLM(L.LightningModule):
#     def __init__(self, model_args, lr=1e-4):
#         super().__init__()
#         self.model = TransformerLM(model_args)
#         self.lr = lr
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         inputs, targets = batch
#         outputs = self(inputs)
#         loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.t().contiguous().view(-1))
#         self.log('train_loss', loss, sync_dist=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         inputs, targets = batch
#         outputs = self(inputs)
#         loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.t().contiguous().view(-1))
#         self.log('val_loss', loss, sync_dist=True)
#         return loss

#     def test_step(self, batch, batch_idx):
#         inputs, targets = batch
#         outputs = self(inputs)
#         loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.t().contiguous().view(-1))
#         self.log('test_loss', loss, sync_dist=True)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         return optimizer



class TransformerBlock(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.attention = nn.MultiheadAttention(model_args.dim, model_args.n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(model_args.dim, 4 * model_args.dim),
            nn.ReLU(),
            nn.Linear(4 * model_args.dim, model_args.dim),
        )
        self.norm1 = nn.LayerNorm(model_args.dim)
        self.norm2 = nn.LayerNorm(model_args.dim)
        
    def forward(self, x: torch.Tensor):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class Transformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        # Using a ModuleDict lets us delete layers without affecting names,
        # ensuring checkpoints will correctly save and load.
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(model_args)
        self.norm = nn.LayerNorm(model_args.dim)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size)

    def forward(self, x):
        # Handling layers being 'None' at runtime enables easy pipeline splitting
        h = self.tok_embeddings(x) if self.tok_embeddings else x
        for layer in self.layers.values():
            h = layer(h)
        h = self.norm(h) if self.norm else h
        output = self.output(h).float() if self.output else h
        return output