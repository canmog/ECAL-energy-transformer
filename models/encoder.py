"""Pre-LN Transformer blocks using F.scaled_dot_product_attention (Flash / mem-
efficient kernels — fast on the RTX 5090 / Blackwell under bf16).

attn_mask convention (SDPA bool): True = participate, False = masked. We pass the
per-key validity mask broadcast as (B, 1, 1, L)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, x, key_valid):
        B, L, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)        # each (B, nh, L, hd)
        attn_mask = key_valid[:, None, None, :]               # (B,1,1,L) bool, True=keep
        p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p)
        out = out.transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.proj(out))
        x = x + self.ff(self.ln2(x))
        return x


class BlockStack(nn.Module):
    def __init__(self, n_blocks, d_model, n_heads, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, ffn_mult, dropout) for _ in range(n_blocks)])

    def forward(self, x, key_valid):
        for blk in self.blocks:
            x = blk(x, key_valid)
        return x
