import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

# Use F.silu directly as the swish activation function
swish = F.silu


@torch.no_grad()
def variance_scaling_init_(tensor: torch.Tensor, scale: float = 1, mode: str = "fan_avg", distribution: str = "uniform"):
    """
    Initializes a tensor with variance scaling.

    Args:
        tensor: The tensor to initialize.
        scale: Scaling factor for the initialization.
        mode: 'fan_in', 'fan_out', or 'fan_avg'.
        distribution: 'normal' or 'uniform'.
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        scale /= fan_in
    elif mode == "fan_out":
        scale /= fan_out
    else:  # fan_avg
        scale /= (fan_in + fan_out) / 2

    if distribution == "normal":
        std = math.sqrt(scale)
        return tensor.normal_(0, std)
    else:  # uniform
        bound = math.sqrt(3 * scale)
        return tensor.uniform_(-bound, bound)


def conv2d(
    in_channel: int,
    out_channel: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
    scale: float = 1,
    mode: str = "fan_avg",
    k: int = 1, # 'k' parameter from the example, not directly used in original
) -> nn.Conv2d:
    """
    Creates a Conv2d layer with variance scaling initialization.
    """
    conv = nn.Conv2d(
        in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias
    )
    variance_scaling_init_(conv.weight, scale, mode=mode)
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


def linear(in_channel: int, out_channel: int, scale: float = 1, mode: str = "fan_avg") -> nn.Linear:
    """
    Creates a Linear layer with variance scaling initialization.
    """
    lin = nn.Linear(in_channel, out_channel)
    variance_scaling_init_(lin.weight, scale, mode=mode)
    nn.init.zeros_(lin.bias)
    return lin


class Swish(nn.Module):
    """
    Swish (SiLU) activation function as a Module.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return swish(input)


class Upsample(nn.Sequential):
    """
    Upsampling block: Nearest neighbor upsample + Conv2d.
    """
    def __init__(self, channel: int):
        super().__init__(
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv2d(channel, channel, 3, padding=1),
        )


class Downsample(nn.Sequential):
    """
    Downsampling block: Strided Conv2d.
    """
    def __init__(self, channel: int):
        super().__init__(conv2d(channel, channel, 3, stride=2, padding=1))


class ResBlock(nn.Module):
    """
    Convolutional Residual Block. Supports affine or additive time conditioning.
    """
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        time_dim: int,
        use_affine_time: bool = False,
        dropout: float = 0,
        group_norm: int = 32,
    ):
        super().__init__()

        self.use_affine_time = use_affine_time
        time_out_dim = out_channel
        time_scale = 1.0 # Default scale for linear layer

        # If using affine time conditioning, output 2x channels for gamma/beta
        # and disable affine in norm2 (handled by gamma/beta)
        norm_affine = True
        if self.use_affine_time:
            time_out_dim *= 2
            time_scale = 1e-10 # Smaller scale for affine time linear layer
            norm_affine = False

        self.norm1 = nn.GroupNorm(group_norm, in_channel)
        self.activation1 = Swish()
        self.conv1 = conv2d(in_channel, out_channel, 3, padding=1)

        self.time = nn.Sequential(
            Swish(), linear(time_dim, time_out_dim, scale=time_scale)
        )

        self.norm2 = nn.GroupNorm(group_norm, out_channel, affine=norm_affine)
        self.activation2 = Swish()
        self.dropout_layer = nn.Dropout(dropout) # Renamed to avoid conflict with forward param
        self.conv2 = conv2d(out_channel, out_channel, 3, padding=1, scale=1e-10)

        self.skip: Optional[nn.Conv2d] = None
        if in_channel != out_channel:
            self.skip = conv2d(in_channel, out_channel, 1)

    def forward(self, input: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        batch = input.shape[0]

        # First block: Norm -> Activation -> Conv
        out = self.conv1(self.activation1(self.norm1(input)))

        # Time conditioning
        time_output = self.time(time).view(batch, -1, 1, 1) # Reshape time for broadcasting
        if self.use_affine_time:
            gamma, beta = time_output.chunk(2, dim=1)
            out = (1 + gamma) * self.norm2(out) + beta
        else: # Additive conditioning
            out = out + time_output
            out = self.norm2(out)

        # Second block: Activation -> Dropout -> Conv
        out = self.conv2(self.dropout_layer(self.activation2(out)))

        # Skip connection
        if self.skip is not None:
            input = self.skip(input)

        return out + input


class SelfAttention(nn.Module):
    """
    Self-attention block.
    """
    def __init__(self, in_channel: int, n_head: int = 1, group_norm: int = 32):
        super().__init__()

        self.n_head = n_head
        self.norm = nn.GroupNorm(group_norm, in_channel)
        self.qkv = conv2d(in_channel, in_channel * 3, 1) # Combined QKV projection
        self.out_conv = conv2d(in_channel, in_channel, 1, scale=1e-10) # Output projection

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head # Head dimension based on total channel and num heads

        norm_input = self.norm(input)
        # Project and split into Q, K, V
        # qkv shape: (B, 3*C, H, W) -> view to (B, n_head, 3*head_dim, H, W)
        qkv = self.qkv(norm_input).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # Each: (B, n_head, head_dim, H, W)

        # Spatial attention: each (h,w) attends to all (y,x)
        # (B, n_head, head_dim, H, W) x (B, n_head, head_dim, Y, X)
        # -> (B, n_head, H_q, W_q, Y_k, X_k)
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(head_dim) # Scale by sqrt(head_dim)
        attn = attn.view(batch, n_head, height, width, -1) # Flatten last two spatial dims for softmax
        attn = torch.softmax(attn, dim=-1) # Apply softmax across attention targets
        attn = attn.view(batch, n_head, height, width, height, width) # Reshape back for einsum with value

        # Weighted sum of values based on attention scores
        # (B, n_head, H_q, W_q, H_k, W_k) x (B, n_head, head_dim, H_v, W_v)
        # -> (B, n_head, head_dim, H_q, W_q)
        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out_conv(out.view(batch, channel, height, width)) # Project back to original channel dim

        return out + input # Residual connection


class TimeEmbedding(nn.Module):
    """
    Positional time embedding layer.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Calculate inverse frequencies for sinusoidal embeddings
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq) # Register as buffer for non-trainable state

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Input 'input' is typically a 1D tensor of timesteps
        shape = input.shape
        # Compute outer product for sinusoid arguments
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        # Concatenate sine and cosine components
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        # Reshape to (batch_size, embedding_dim)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class ResBlockWithAttention(nn.Module):
    """
    Combines a ResBlock and an optional SelfAttention block.
    """
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        time_dim: int,
        dropout: float,
        use_attention: bool = False,
        attention_head: int = 1,
        use_affine_time: bool = False,
        group_norm: int = 32,
    ):
        super().__init__()

        self.resblock = ResBlock(
            in_channel, out_channel, time_dim, use_affine_time, dropout, group_norm=group_norm
        )

        self.attention: Optional[SelfAttention] = None
        if use_attention:
            self.attention = SelfAttention(out_channel, n_head=attention_head, group_norm=group_norm)

    def forward(self, input: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        out = self.resblock(input, time)
        if self.attention is not None:
            out = self.attention(out)
        return out


def spatial_fold(input: torch.Tensor, fold: int) -> torch.Tensor:
    """
    Folds spatial dimensions into the channel dimension.
    Used for creating a 'folded' input, e.g., for multi-channel processing or different input formats.
    """
    if fold == 1:
        return input

    batch, channel, height, width = input.shape
    h_fold = height // fold
    w_fold = width // fold

    # Reshape (B, C, H, W) -> (B, C, h_fold, fold, w_fold, fold)
    # Permute to (B, C, fold, fold, h_fold, w_fold)
    # Reshape to (B, C * fold * fold, h_fold, w_fold)
    return (
        input.view(batch, channel, h_fold, fold, w_fold, fold)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch, -1, h_fold, w_fold)
    )


def spatial_unfold(input: torch.Tensor, unfold: int) -> torch.Tensor:
    """
    Unfolds channels back into spatial dimensions.
    Reverse operation of spatial_fold.
    """
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    h_unfold = height * unfold
    w_unfold = width * unfold

    # Reshape (B, C, H, W) -> (B, C_orig, unfold, unfold, H, W)
    # where C_orig = C / (unfold * unfold)
    # Permute to (B, C_orig, H, unfold, W, unfold)
    # Reshape to (B, C_orig, H*unfold, W*unfold)
    return (
        input.view(batch, -1, unfold, unfold, height, width)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(batch, -1, h_unfold, w_unfold)
    )


class UNet_final(nn.Module):
    """
    A UNet architecture for diffusion models, styled similarly to
    the rosinality/denoising-diffusion-pytorch implementation.
    """
    def __init__(
        self,
        in_channel: int, # Input image channels (e.g., 3 for RGB)
        channel: int, # Base channel dimension for the UNet
        channel_multiplier: List[int], # Multiplier for channels at each UNet level
        n_res_blocks: int, # Number of ResBlocks per level
        attn_strides: List[int], # Spatial strides at which attention blocks are applied (e.g., [4, 8, 16])
        attn_heads: int = 1, # Number of attention heads
        use_affine_time: bool = False, # Whether to use affine transformation for time conditioning
        dropout: float = 0, # Dropout rate
        fold: int = 1, # Spatial folding factor
        k: int = 1, # 'k' parameter, likely for multiple output heads (e.g., for different noise prediction tasks)
    ):
        super().__init__()

        self.fold = fold
        self.k = k
        time_dim = channel * 4 # Dimension of the expanded time embedding
        group_norm = channel // 4 # Number of groups for GroupNorm

        n_block = len(channel_multiplier)

        # Time embedding pathway
        self.time = nn.Sequential(
            TimeEmbedding(channel), # Initial sinusoidal embedding
            linear(channel, time_dim), # First linear projection
            Swish(), # Swish activation
            linear(time_dim, time_dim), # Second linear projection
        )

        # Downsampling path layers
        down_layers: List[nn.Module] = [conv2d(in_channel * (fold ** 2), channel, 3, padding=1)] # Initial convolution
        feat_channels = [channel] # Stores channel counts for skip connections
        current_in_channel = channel

        for i in range(n_block):
            for _ in range(n_res_blocks):
                channel_mult = channel * channel_multiplier[i]
                down_layers.append(
                    ResBlockWithAttention(
                        current_in_channel,
                        channel_mult,
                        time_dim,
                        dropout,
                        use_attention=(2 ** i in attn_strides), # Apply attention based on resolution
                        attention_head=attn_heads,
                        use_affine_time=use_affine_time,
                        group_norm=group_norm,
                    )
                )
                feat_channels.append(channel_mult)
                current_in_channel = channel_mult # Update current_in_channel for next block

            # Downsample between levels (except for the last level)
            if i != n_block - 1:
                down_layers.append(Downsample(current_in_channel))
                feat_channels.append(current_in_channel) # Add channels after downsample for skip if needed

        self.down = nn.ModuleList(down_layers)

        # Middle (bottleneck) layers
        self.mid = nn.ModuleList(
            [
                ResBlockWithAttention( # First middle block (with attention)
                    current_in_channel,
                    current_in_channel,
                    time_dim,
                    dropout=dropout,
                    use_attention=True, # Attention always applied in the middle
                    attention_head=attn_heads,
                    use_affine_time=use_affine_time,
                    group_norm=group_norm,
                ),
                ResBlockWithAttention( # Second middle block (without attention)
                    current_in_channel,
                    current_in_channel,
                    time_dim,
                    dropout=dropout,
                    use_attention=False,
                    use_affine_time=use_affine_time,
                    group_norm=group_norm,
                ),
            ]
        )

        # Upsampling path layers
        up_layers: List[nn.Module] = []
        # Iterate in reverse order of downsampling levels
        for i in reversed(range(n_block)):
            # For each level, add (n_res_blocks + 1) ResBlocks (including the one that merges skip)
            for _ in range(n_res_blocks + 1):
                channel_mult = channel * channel_multiplier[i]
                # Pop the last feature map from `feat_channels` for the skip connection
                up_layers.append(
                    ResBlockWithAttention(
                        current_in_channel + feat_channels.pop(), # Input = current_features + skip_features
                        channel_mult,
                        time_dim,
                        dropout=dropout,
                        use_attention=(2 ** i in attn_strides),
                        attention_head=attn_heads,
                        use_affine_time=use_affine_time,
                        group_norm=group_norm,
                    )
                )
                current_in_channel = channel_mult # Update current_in_channel for next block

            # Upsample between levels (except for the first level)
            if i != 0:
                up_layers.append(Upsample(current_in_channel))

        self.up = nn.ModuleList(up_layers)

        # Final output block
        self.out = nn.Sequential(
            nn.GroupNorm(group_norm, current_in_channel), # Final GroupNorm
            Swish(), # Final activation
            conv2d(current_in_channel, 3 * (fold ** 2) * self.k, 3, padding=1, scale=1e-10, k=self.k), # Output convolution
        )

    def forward(self, input: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # Generate time embedding
        time_embed = self.time(time)

        feats: List[torch.Tensor] = [] # To store feature maps for skip connections

        # Spatial folding of the input
        out = spatial_fold(input, self.fold)

        # Downsampling path
        for layer in self.down:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(out, time_embed)
            else: # Downsample layer
                out = layer(out)
            feats.append(out) # Save output of each block for skip connections

        # Middle path
        for layer in self.mid:
            out = layer(out, time_embed)

        # The last feature in `feats` is from the bottleneck of the downsampling path,
        # which is already `out` from the middle layer. So, pop it once.
        # This is because the last `append(out)` in the down loop is the input to `mid`.
        # Ensure `feats` is correctly managed for skip connections.
        # The first element popped from feats should be the one BEFORE the final downsample (if any)
        # and after the last ResBlockWithAttention in the downsampling path.
        # Given the construction, `feats.pop()` will provide the correct skip connection.
        feats.pop() # Remove the last feature that went into the middle block

        # Upsampling path
        for layer in self.up:
            if isinstance(layer, ResBlockWithAttention):
                # Concatenate current output with corresponding skip connection
                out = layer(torch.cat((out, feats.pop()), 1), time_embed)
            else: # Upsample layer
                out = layer(out)

        # Final output layer
        out = self.out(out)

        # Spatial unfolding of the output
        out = spatial_unfold(out, self.fold)

        return out

