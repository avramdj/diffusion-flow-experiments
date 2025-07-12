import numpy as np
import torch
from torch import nn


class Patchify(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.p = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.unfold(x).transpose(1, 2)


class Unpatchify(nn.Module):
    def __init__(self, patch_size: int, img_size: tuple[int, int]):
        super().__init__()
        self.p = patch_size
        self.img_size = img_size
        self.fold = nn.Fold(
            output_size=img_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.fold(x.transpose(1, 2))


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
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cond_proj = nn.Linear(cond_dim, d_model * 3)

        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x, cond):
        gamma, beta, alpha = self.cond_proj(cond).unsqueeze(1).chunk(3, dim=-1)

        modulated = self.norm(x) * (1 + gamma) + beta

        return modulated, alpha


class DiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, cond_dim: int):
        super().__init__()
        self.attn_mod = AdaLNZero(d_model, cond_dim)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.ffn_mod = AdaLNZero(d_model, cond_dim)
        self.ffn = PointwiseFFN(d_model, factor=4)

    def forward(self, x, cond):
        modulated_x, alpha_attn = self.attn_mod(x, cond)
        attn_output = self.attn(modulated_x, modulated_x, modulated_x)[0]
        x = x + alpha_attn * attn_output

        modulated_x, alpha_ffn = self.ffn_mod(x, cond)
        ffn_output = self.ffn(modulated_x)
        x = x + alpha_ffn * ffn_output

        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, proj_dim: int):
        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2

        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(-torch.arange(half_dim, device=device) * emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        return self.mlp(emb)


class DiT(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_size: int,
        img_size: tuple[int, int],
        n_heads: int,
        n_layers: int,
        in_channels: int,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.d_model = d_model
        self.in_channels = in_channels
        self.n_heads = n_heads

        self.time_embed = TimestepEmbedding(dim=64, proj_dim=self.d_model)

        self.patchify = Patchify(patch_size=patch_size)
        patch_dim = self.in_channels * self.patch_size * self.patch_size
        self.patch_proj = nn.Linear(patch_dim, self.d_model)

        self.num_patches = (self.img_size[0] // self.patch_size) * (
            self.img_size[1] // self.patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))

        self.layers = nn.ModuleList(
            [
                DiTBlock(self.d_model, self.n_heads, self.d_model)
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(self.d_model, elementwise_affine=False)
        self.final_proj = nn.Linear(self.d_model, patch_dim)

        self.unpatchify = Unpatchify(self.patch_size, self.img_size)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = self.get_2d_sincos_pos_embed(
            self.d_model, int(self.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, x, t):
        cond = self.time_embed(t)

        x = self.patchify(x)
        x = self.patch_proj(x)

        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x, cond)

        x = self.final_norm(x)
        x = self.final_proj(x)

        return self.unpatchify(x)

    def ode_forward(self, t: torch.Tensor, x: torch.Tensor):
        t_batch = t.repeat(x.shape[0])
        return self.forward(x, t_batch)

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        return self.get_pos_embed_from_grid(embed_dim, grid)

    def get_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = self.get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
        emb_w = self.get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
        return np.concatenate([emb_h, emb_w], axis=1)

    def get_1d_sincos_pos_embed(self, embed_dim, pos):
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        return np.concatenate([emb_sin, emb_cos], axis=1)
