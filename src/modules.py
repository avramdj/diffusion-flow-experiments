import numpy as np
import torch
from torch import nn


class Patchify(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.p = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        bsz, c, h, w = x.shape
        assert h % self.p == 0 and w % self.p == 0, (
            f"Image size {h}x{w} must be divisible by patch_size {self.p}"
        )
        x = self.unfold(x)  # shape [bsz, c * p * p, L]
        x = x.transpose(1, 2)  # shape [bsz, L, c * p * p]
        return x


class Unpatchify(nn.Module):
    def __init__(self, patch_size: int, img_size: tuple[int, int]):
        super().__init__()
        self.p = patch_size
        self.img_size = img_size
        self.fold = nn.Fold(
            output_size=img_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        bsz, L, patch_dim = x.shape
        H, W = self.img_size
        p = self.p
        # infer c
        assert H % p == 0 and W % p == 0, (
            f"Image size {(H, W)} must be divisible by patch size {p}"
        )
        expected_L = (H // p) * (W // p)
        assert L == expected_L, (
            f"L={L} does not match expected {(H // p)}*{(W // p)}={expected_L}"
        )
        # patch_dim should be c*p*p
        c_times_p2 = patch_dim
        assert c_times_p2 % (p * p) == 0, (
            f"patch_dim={patch_dim} is not divisible by p*p={p * p}"
        )
        x = x.transpose(1, 2)  # [b, patch_dim, L]
        recon = self.fold(x)  # [b, c, H, W]
        return recon


class PointwiseFFN(nn.Module):
    def __init__(self, d_model: int, factor: int = 4):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * factor),
            nn.SiLU(),
            nn.Linear(d_model * factor, d_model),
        )

    def forward(self, x):
        return self.ffn(x)


class AdaLNZero(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim * 3)

        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x, cond):
        # bsz, seqlen, d_model = x.shape
        # cond_proj -> gamma, beta, alpha (all shaped (B, D))
        gamma_beta_alpha = self.cond_proj(cond)  # (B, 3*D)
        gamma, beta, alpha = gamma_beta_alpha.chunk(3, dim=-1)  # each (B, D)

        # reshape to (B, 1, D) to broadcast across T
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        alpha = alpha.unsqueeze(1)

        x_norm = self.norm(x)  # (bsz, seqlen, d_model)
        out = x_norm * (1 + gamma) + beta
        return out, alpha


class DiTBlockWithCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.pre_sa_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)
        self.pre_ca_ln = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=n_heads, batch_first=True
        )
        self.pre_ffn_ln = nn.LayerNorm(d_model)
        self.ffn = PointwiseFFN(d_model, factor=4)

    def forward(self, x, y):
        residual = x
        x = self.pre_sa_ln(x)
        x, _ = self.attn(x, x, x)
        x = x + residual
        residual = x
        x = self.pre_ca_ln(x)
        x, _ = self.cross_attn(x, y, y)
        x = x + residual
        residual = x
        x = self.pre_ffn_ln(x)
        x = self.ffn(x)
        x = x + residual
        return x


class DiTBlockWithAdaLN(nn.Module):
    def __init__(self, d_model: int, n_heads: int, cond_dim: int):
        super().__init__()
        self.pre_sa_ln = AdaLNZero(d_model, cond_dim)
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)
        self.pre_ffn_ln = AdaLNZero(d_model, cond_dim)
        self.ffn = PointwiseFFN(d_model, factor=4)

    def forward(self, x, cond):
        residual = x
        x, alpha = self.pre_sa_ln(x, cond)
        x, _ = self.attn(x, x, x)
        x = x * alpha
        x = x + residual
        residual = x
        x, alpha = self.pre_ffn_ln(x, cond)
        x = self.ffn(x)
        x = x * alpha
        x = x + residual
        return x


class DiT(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_size: int,
        img_size: tuple[int, int],
        n_heads: int,
        cond_dim: int,
        n_layers: int,
        in_channels: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.d_model = d_model
        self.in_channels = in_channels
        self.patchify = Patchify(patch_size=patch_size)

        patch_dim = self.in_channels * self.patch_size * self.patch_size
        self.patch_proj = nn.Linear(patch_dim, self.d_model)

        self.num_patches = (self.img_size[0] // self.patch_size) * (
            self.img_size[1] // self.patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))

        self.layers = nn.ModuleList(
            [DiTBlockWithAdaLN(d_model, n_heads, cond_dim) for _ in range(n_layers)]
        )

        self.final_ln = nn.LayerNorm(self.d_model, elementwise_affine=False)
        self.final_proj = nn.Linear(self.d_model, patch_dim)
        self.unpatchify = Unpatchify(self.patch_size, self.img_size)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = self.get_2d_sincos_pos_embed(
            self.d_model, int(self.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.zeros_(self.final_proj.weight)
        torch.nn.init.zeros_(self.final_proj.bias)

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size, cls_token=False):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, grid[0]
        )  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, grid[1]
        )  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def forward(self, x, cond):
        x = self.patchify(x)
        x = self.patch_proj(x)
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x, cond)
        x = self.final_ln(x)
        x = self.final_proj(x)
        x = self.unpatchify(x)
        return x
