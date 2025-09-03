import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


def get_timestep_embedding(timesteps, embedding_dim, max_time=1000., dtype=torch.float32):
    """Build sinusoidal embeddings (from Fairseq)."""
    assert len(timesteps.shape) == 1
    timesteps = timesteps * (1000. / max_time)

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0), value=0.0)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def nearest_neighbor_upsample(x):
    """Upsample using nearest neighbor interpolation."""
    return F.interpolate(x, scale_factor=2, mode="nearest")


class Normalize(nn.GroupNorm):
    """Group normalization with default settings."""
    def __init__(self, num_channels):
        super().__init__(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class ResidualBlock(nn.Module):
    """Convolutional residual block."""
    def __init__(self, in_ch, out_ch=None, resample=None, dropout=0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch if out_ch is not None else in_ch
        self.resample = resample
        
        self.norm1 = Normalize(in_ch)
        self.silu = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, self.out_ch, 3, padding=1, stride=1)
        
        # Time embedding projection
        self.time_emb = nn.Linear(in_ch, 2 * self.out_ch)  # Should match emb_ch dimension
        
        self.norm2 = Normalize(self.out_ch)
        self.dropout_layer = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_ch, self.out_ch, 3, padding=1, stride=1)
        
        # Initialize conv2 to zeros like in JAX version
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        # Shortcut connection if channels change
        if in_ch != self.out_ch:
            self.shortcut = nn.Conv2d(in_ch, self.out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb, deterministic=True):
        # x: (B, C, H, W), emb: (B, emb_ch)
        B, C, H, W = x.shape
        assert emb.shape[0] == B and len(emb.shape) == 2
        
        h = self.norm1(x)
        h = self.silu(h)
        
        # Handle resampling
        if self.resample == 'up':
            h = nearest_neighbor_upsample(h)
            x = nearest_neighbor_upsample(x)
        elif self.resample == 'down':
            h = F.avg_pool2d(h, 2, stride=2)
            x = F.avg_pool2d(x, 2, stride=2)
            
        h = self.conv1(h)
        
        # Add timestep embedding
        emb_out = self.silu(emb)
        emb_out = self.time_emb(emb_out)  # (B, 2*out_ch)
        scale, shift = torch.chunk(emb_out, 2, dim=1)  # Each (B, out_ch)
        scale = scale[:, :, None, None]  # (B, out_ch, 1, 1)
        shift = shift[:, :, None, None]  # (B, out_ch, 1, 1)
        
        h = self.norm2(h) * (1 + scale) + shift
        h = self.silu(h)
        h = self.dropout_layer(h) if not deterministic else h
        h = self.conv2(h)
        
        # Shortcut connection
        x = self.shortcut(x)
        
        return x + h


class AttentionBlock(nn.Module):
    """Self-attention residual block."""
    def __init__(self, in_ch, num_heads=None, head_dim=None):
        super().__init__()
        self.in_ch = in_ch
        
        if head_dim is None:
            assert num_heads is not None
            assert in_ch % num_heads == 0
            self.num_heads = num_heads
            self.head_dim = in_ch // num_heads
        else:
            assert num_heads is None
            assert in_ch % head_dim == 0
            self.head_dim = head_dim
            self.num_heads = in_ch // head_dim
            
        self.norm = Normalize(in_ch)
        self.q = nn.Linear(in_ch, in_ch)
        self.k = nn.Linear(in_ch, in_ch)
        self.v = nn.Linear(in_ch, in_ch)
        self.proj_out = nn.Linear(in_ch, in_ch)
        
        # Initialize proj_out to zeros like in JAX version
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert C == self.in_ch
        
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        
        q = self.q(h).view(B, H * W, self.num_heads, self.head_dim)
        k = self.k(h).view(B, H * W, self.num_heads, self.head_dim)
        v = self.v(h).view(B, H * W, self.num_heads, self.head_dim)
        
        # Transpose for attention: (B, num_heads, H*W, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        h = F.scaled_dot_product_attention(q, k, v)
        
        # Back to (B, H*W, num_heads, head_dim)
        h = h.permute(0, 2, 1, 3)
        h = h.reshape(B, H * W, C)
        
        h = self.proj_out(h)
        h = h.permute(0, 2, 1).view(B, C, H, W)  # Back to (B, C, H, W)
        
        return x + h


class UNet(nn.Module):
    def __init__(self, num_classes, ch, emb_ch, out_ch, ch_mult, num_res_blocks,
                 attn_resolutions, num_heads=None, head_dim=None, dropout=0.0,
                 logsnr_input_type="linear", logsnr_scale_range=(-10., 10.),
                 resblock_resample=False):
        super().__init__()
        self.num_classes = num_classes
        self.ch = ch
        self.emb_ch = emb_ch
        self.out_ch = out_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.logsnr_input_type = logsnr_input_type
        self.logsnr_scale_range = logsnr_scale_range
        self.resblock_resample = resblock_resample
        
        num_resolutions = len(ch_mult)
        
        # Time embedding layers
        self.time_dense0 = nn.Linear(ch, emb_ch)
        self.time_dense1 = nn.Linear(emb_ch, emb_ch)
        
        # Class embedding
        if num_classes > 1:
            self.class_emb = nn.Linear(num_classes, emb_ch)
        
        # Initial conv
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1, stride=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        in_ch = ch
        for i_level in range(num_resolutions):
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            
            for i_block in range(num_res_blocks):
                out_ch_level = ch * ch_mult[i_level]
                level_blocks.append(ResidualBlock(in_ch, out_ch_level, dropout=dropout))
                
                if out_ch_level in attn_resolutions:
                    level_attns.append(AttentionBlock(out_ch_level, num_heads, head_dim))
                else:
                    level_attns.append(nn.Identity())
                    
                in_ch = out_ch_level
            
            self.down_blocks.append(level_blocks)
            self.down_attns.append(level_attns)
            
            # Downsample (except for last level)
            if i_level != num_resolutions - 1:
                if resblock_resample:
                    self.down_samples.append(ResidualBlock(in_ch, in_ch, resample='down', dropout=dropout))
                else:
                    self.down_samples.append(nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1))
            else:
                self.down_samples.append(nn.Identity())
        
        # Middle blocks
        self.mid_block1 = ResidualBlock(in_ch, dropout=dropout)
        self.mid_attn = AttentionBlock(in_ch, num_heads, head_dim)
        self.mid_block2 = ResidualBlock(in_ch, dropout=dropout)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i_level in reversed(range(num_resolutions)):
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            
            for i_block in range(num_res_blocks + 1):
                out_ch_level = ch * ch_mult[i_level]
                # First block takes concatenated input
                if i_block == 0:
                    level_blocks.append(ResidualBlock(in_ch + out_ch_level, out_ch_level, dropout=dropout))
                else:
                    level_blocks.append(ResidualBlock(out_ch_level, out_ch_level, dropout=dropout))
                
                if out_ch_level in attn_resolutions:
                    level_attns.append(AttentionBlock(out_ch_level, num_heads, head_dim))
                else:
                    level_attns.append(nn.Identity())
                    
                in_ch = out_ch_level
            
            self.up_blocks.append(level_blocks)
            self.up_attns.append(level_attns)
            
            # Upsample (except for first level when going in reverse)
            if i_level != 0:
                if resblock_resample:
                    self.up_samples.append(ResidualBlock(in_ch, in_ch, resample='up', dropout=dropout))
                else:
                    self.up_samples.append(nn.ModuleList([
                        lambda x: nearest_neighbor_upsample(x),
                        nn.Conv2d(in_ch, in_ch, 3, padding=1, stride=1)
                    ]))
            else:
                self.up_samples.append(nn.Identity())
        
        # Final layers
        self.norm_out = Normalize(ch)
        self.conv_out = nn.Conv2d(ch, self.out_ch, 3, padding=1, stride=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, logsnr, y =1, train=False):
        B, C, H, W = x.shape
        assert H == W
        assert logsnr.shape == (B,)
        
        # Timestep embedding
        if self.logsnr_input_type == 'linear':
            logsnr_input = (logsnr - self.logsnr_scale_range[0]) / (
                self.logsnr_scale_range[1] - self.logsnr_scale_range[0])
        elif self.logsnr_input_type == 'sigmoid':
            logsnr_input = torch.sigmoid(logsnr)
        elif self.logsnr_input_type == 'inv_cos':
            logsnr_input = (torch.atan(torch.exp(-0.5 * torch.clamp(logsnr, -20., 20.)))
                           / (0.5 * math.pi))
        else:
            raise NotImplementedError(self.logsnr_input_type)
        
        emb = get_timestep_embedding(logsnr_input, self.ch, max_time=1.)
        emb = self.time_dense0(emb)
        emb = self.time_dense1(F.silu(emb))
        
        # Class embedding
        if self.num_classes > 1:
            assert y.shape == (B,) and y.dtype == torch.long
            y_emb = F.one_hot(y, num_classes=self.num_classes).float()
            y_emb = self.class_emb(y_emb)
            emb = emb + y_emb
        
        # Initial conv
        h = self.conv_in(x)
        hs = [h]
        
        # Downsampling
        for i_level in range(len(self.ch_mult)):
            for i_block in range(self.num_res_blocks):
                h = self.down_blocks[i_level][i_block](h, emb, deterministic=not train)
                h = self.down_attns[i_level][i_block](h)
                hs.append(h)
            
            # Downsample
            if i_level != len(self.ch_mult) - 1:
                if self.resblock_resample:
                    h = self.down_samples[i_level](h, emb, deterministic=not train)
                else:
                    h = self.down_samples[i_level](h)
                hs.append(h)
        
        # Middle``
        h = self.mid_block1(h, emb, deterministic=not train)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb, deterministic=not train)
        
        # Upsampling
        for i_level in reversed(range(len(self.ch_mult))):
            for i_block in range(self.num_res_blocks + 1):
                # Concatenate with skip connection
                h = torch.cat([h, hs.pop()], dim=1)
                h = self.up_blocks[len(self.ch_mult) - 1 - i_level][i_block](h, emb, deterministic=not train)
                h = self.up_attns[len(self.ch_mult) - 1 - i_level][i_block](h)
            
            # Upsample
            if i_level != 0:
                if self.resblock_resample:
                    h = self.up_samples[len(self.ch_mult) - 1 - i_level](h, emb, deterministic=not train)
                else:
                    up_modules = self.up_samples[len(self.ch_mult) - 1 - i_level]
                    if isinstance(up_modules, nn.ModuleList):
                        h = up_modules[0](h)  # upsample
                        h = up_modules[1](h)  # conv
                    # If it's Identity, do nothing
        
        # Final
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        
        return h
    


# model_config = dict(
#     num_classes=1,
#     ch=256,
#     emb_ch=1024,
#     out_ch=6,  # e.g., RGB output
#     ch_mult=(1, 1, 1),
#     num_res_blocks=3,
#     attn_resolutions=(8, 16),
#     num_heads=1,
#     dropout=0.2,
#     logsnr_input_type="inv_cos",
#     resblock_resample=True,
# )

# import torch
# from unet_jax1 import UNet  # your UNet implementation

# # UNet config
# model_config = dict(
#     num_classes=1,
#     ch=256,
#     emb_ch=1024,
#     out_ch=6,
#     ch_mult=(1, 1, 1),
#     num_res_blocks=3,
#     attn_resolutions=(8, 16),
#     num_heads=1,
#     dropout=0.2,
#     logsnr_input_type="inv_cos",
#     resblock_resample=True,
# )

# # Instantiate model
# model = UNet(**model_config)

# # Function to count parameters
# def count_params(model):
#     return sum(p.numel() for p in model.parameters())

# print("Total parameters:", count_params(model))
