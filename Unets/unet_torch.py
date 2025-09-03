import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

def get_timestep_embedding(timesteps, embedding_dim, max_time=1000., dtype=torch.float32):
    """Build sinusoidal embeddings (from Fairseq).

    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".

    Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        max_time: float: largest time input
        dtype: data type of the generated embeddings

    Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(timesteps.shape) == 1
    timesteps = timesteps * (1000. / max_time)

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=timesteps.device) * -emb)
    emb = timesteps.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def nearest_neighbor_upsample(x):
    """Nearest neighbor upsampling by factor of 2."""
    return F.interpolate(x, scale_factor=2, mode='nearest')


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    """Convolutional residual block."""

    def __init__(self, in_ch, out_ch=None, dropout=0.0, resample=None, emb_ch=1024):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch if out_ch is not None else in_ch
        self.dropout = dropout
        self.resample = resample
        
        # Ensure group norm works with any number of channels
        # Use min(32, in_ch) to handle cases where in_ch < 32
        groups1 = min(32, in_ch)
        while in_ch % groups1 != 0 and groups1 > 1:
            groups1 -= 1
        self.norm1 = nn.GroupNorm(groups1, in_ch)
        
        self.conv1 = nn.Conv2d(in_ch, self.out_ch, 3, padding=1)
        
        # Time embedding projection - make emb_ch configurable
        self.temb_proj = nn.Linear(emb_ch, 2 * self.out_ch)
        
        # Same logic for second norm
        groups2 = min(32, self.out_ch)
        while self.out_ch % groups2 != 0 and groups2 > 1:
            groups2 -= 1
        self.norm2 = nn.GroupNorm(groups2, self.out_ch)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_ch, self.out_ch, 3, padding=1)
        
        # Initialize conv2 to zeros
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        # Shortcut connection
        if in_ch != self.out_ch:
            self.nin_shortcut = nn.Conv2d(in_ch, self.out_ch, 1)
        else:
            self.nin_shortcut = nn.Identity()
            
        self.swish = Swish()

    def forward(self, x, emb):
        B, C, H, W = x.shape
        assert emb.shape[0] == B and len(emb.shape) == 2
        
        h = self.swish(self.norm1(x))
        
        # Handle resampling
        if self.resample == 'up':
            h = nearest_neighbor_upsample(h)
            x = nearest_neighbor_upsample(x)
        elif self.resample == 'down':
            h = F.avg_pool2d(h, 2, 2)
            x = F.avg_pool2d(x, 2, 2)
            
        h = self.conv1(h)
        
        # Add in timestep embedding
        emb_out = self.temb_proj(self.swish(emb))[:, :, None, None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        
        # Rest of the block
        h = self.swish(h)
        h = self.dropout_layer(h)
        h = self.conv2(h)
        
        # Shortcut connection
        x = self.nin_shortcut(x)
        
        assert x.shape == h.shape
        return x + h


class AttnBlock(nn.Module):
    """Self-attention residual block."""

    def __init__(self, ch, num_heads=None, head_dim=None):
        super().__init__()
        self.ch = ch
        
        if head_dim is None:
            assert num_heads is not None
            assert ch % num_heads == 0
            self.num_heads = num_heads
            self.head_dim = ch // num_heads
        else:
            assert num_heads is None
            assert ch % head_dim == 0
            self.head_dim = head_dim
            self.num_heads = ch // head_dim
        
        # Ensure group norm works with any number of channels
        groups = min(32, ch)
        while ch % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, ch)
        
        self.q = nn.Linear(ch, ch)
        self.k = nn.Linear(ch, ch)
        self.v = nn.Linear(ch, ch)
        self.proj_out = nn.Linear(ch, ch)
        
        # Initialize proj_out to zeros
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        
        h = self.norm(x)
        h = h.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        
        q = self.q(h).view(B, H * W, self.num_heads, self.head_dim)
        k = self.k(h).view(B, H * W, self.num_heads, self.head_dim)
        v = self.v(h).view(B, H * W, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * scale
        attn = F.softmax(attn, dim=2)
        h = torch.einsum('bijh,bjhd->bihd', attn, v)
        
        h = h.view(B, H * W, C)
        h = self.proj_out(h)
        h = h.transpose(1, 2).view(B, C, H, W)
        
        return x + h


class UpsampleModule(nn.Module):
    """Proper upsample module."""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    
    def forward(self, x):
        x = nearest_neighbor_upsample(x)
        return self.conv(x)


class UNet(nn.Module):
    """A UNet architecture."""

    def __init__(self, 
                 num_classes=1,
                 ch=256,
                 emb_ch=1024,
                 out_ch=3,
                 ch_mult=(1, 1, 1),
                 num_res_blocks=3,
                 attn_resolutions=(8, 16),
                 num_heads=1,
                 dropout=0.2,
                 logsnr_input_type='inv_cos',
                 logsnr_scale_range=(-10., 10.),
                 resblock_resample=False,
                 head_dim=None,
                 input_size=32):
        super().__init__()
        
        self.num_classes = num_classes
        self.ch = ch
        self.emb_ch = emb_ch
        self.out_ch = out_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.logsnr_input_type = logsnr_input_type
        self.logsnr_scale_range = logsnr_scale_range
        self.resblock_resample = resblock_resample
        self.head_dim = head_dim
        self.input_size = input_size
        
        num_resolutions = len(ch_mult)
        
        # Time embedding layers
        self.dense0 = nn.Linear(ch, emb_ch)
        self.dense1 = nn.Linear(emb_ch, emb_ch)
        
        # Class embedding (if conditional)
        if num_classes > 1:
            self.class_emb = nn.Linear(num_classes, emb_ch)
        
        self.swish = Swish()
        
        # Initial convolution
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)
        
        # Build the network architecture
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        
        # Keep track of all channel dimensions for skip connections
        skip_channels = [ch]  # Start with initial conv output
        current_ch = ch
        
        for i_level in range(num_resolutions):
            blocks = nn.ModuleList()
            out_ch_level = ch * ch_mult[i_level]
            
            for i_block in range(num_res_blocks):
                blocks.append(ResnetBlock(current_ch, out_ch_level, dropout, emb_ch=emb_ch))
                current_ch = out_ch_level
                skip_channels.append(current_ch)
                
                # Add attention if at specified resolution
                current_res = self.input_size // (2 ** i_level)
                if current_res in attn_resolutions:
                    blocks.append(AttnBlock(current_ch, num_heads, head_dim))
            
            self.down_blocks.append(blocks)
            
            # Downsample (except for last level)
            if i_level != num_resolutions - 1:
                if resblock_resample:
                    downsample = ResnetBlock(current_ch, current_ch, dropout, 
                                           resample='down', emb_ch=emb_ch)
                else:
                    downsample = nn.Conv2d(current_ch, current_ch, 3, stride=2, padding=1)
                self.down_sample.append(downsample)
                skip_channels.append(current_ch)
            else:
                self.down_sample.append(None)
        
        # Middle blocks
        mid_ch = current_ch
        self.mid_block1 = ResnetBlock(mid_ch, mid_ch, dropout, emb_ch=emb_ch)
        self.mid_attn = AttnBlock(mid_ch, num_heads, head_dim)
        self.mid_block2 = ResnetBlock(mid_ch, mid_ch, dropout, emb_ch=emb_ch)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        
        # Reverse the skip_channels for upsampling
        skip_channels_up = skip_channels[::-1]  # Make a copy and reverse
        skip_idx = 0
        
        for i_level in reversed(range(num_resolutions)):
            # Upsample first (except for the deepest level)
            if i_level != num_resolutions - 1:
                if resblock_resample:
                    upsample = ResnetBlock(current_ch, current_ch, dropout, 
                                         resample='up', emb_ch=emb_ch)
                else:
                    upsample = UpsampleModule(current_ch)
                self.up_sample.append(upsample)
            else:
                self.up_sample.append(None)
            
            blocks = nn.ModuleList()
            out_ch_level = ch * ch_mult[i_level]
            
            for i_block in range(num_res_blocks + 1):
                # Calculate input channels (current + skip connection)
                skip_ch = skip_channels_up[skip_idx]
                skip_idx += 1
                in_ch_block = current_ch + skip_ch
                
                blocks.append(ResnetBlock(in_ch_block, out_ch_level, dropout, emb_ch=emb_ch))
                current_ch = out_ch_level
                
                # Add attention if at specified resolution
                current_res = self.input_size // (2 ** i_level)
                if current_res in attn_resolutions:
                    blocks.append(AttnBlock(current_ch, num_heads, head_dim))
            
            self.up_blocks.append(blocks)
        
        # Output layers
        groups_out = min(32, ch)
        while ch % groups_out != 0 and groups_out > 1:
            groups_out -= 1
        self.norm_out = nn.GroupNorm(groups_out, ch)
        self.conv_out = nn.Conv2d(ch, out_ch, 3, padding=1)
        
        # Initialize conv_out to zeros
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, logsnr, y=None, train=True):
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
        
        emb = get_timestep_embedding(logsnr_input, embedding_dim=self.ch, max_time=1.)
        emb = self.dense0(emb)
        emb = self.dense1(self.swish(emb))
        
        # Class embedding
        if self.num_classes > 1:
            assert y is not None and y.shape == (B,)
            y_emb = F.one_hot(y, num_classes=self.num_classes).float()
            y_emb = self.class_emb(y_emb)
            emb = emb + y_emb
        
        # Initial convolution
        h = self.conv_in(x)
        hs = [h]
        # Downsampling
        
        for i_level in range(len(self.ch_mult)):
            # ResNet blocks
            for block in self.down_blocks[i_level]:
                if isinstance(block, ResnetBlock):
                    h = block(h, emb)
                elif isinstance(block, AttnBlock):
                    h = block(h)
                hs.append(h)
            
            # Downsample
            if self.down_sample[i_level] is not None:
                if self.resblock_resample:
                    h = self.down_sample[i_level](h, emb)
                else:
                    h = self.down_sample[i_level](h)
                hs.append(h)
        # Middle
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)
        # Upsampling
        for i_level in range(len(self.ch_mult)):
            for block in self.up_blocks[i_level]:
                if isinstance(block, ResnetBlock):
                    skip = hs.pop()
                    if h.shape[2:] != skip.shape[2:]:
                        h = F.interpolate(h, size=skip.shape[2:], mode='nearest')
                    h = torch.cat([h, skip], dim=1)
                    h = block(h, emb)
                elif isinstance(block, AttnBlock):
                    h = block(h)
            
            # Upsample
            if self.up_sample[i_level] is not None:
                if self.resblock_resample:
                    h = self.up_sample[i_level](h, emb)
                else:
                    h = self.up_sample[i_level](h)

        # Output
        h = self.swish(self.norm_out(h))
        h = self.conv_out(h)
        return h