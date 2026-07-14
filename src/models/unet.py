"""
U-Net Architecture for Diffusion Models

In this file, you should implements a U-Net architecture suitable for DDPM.

Architecture Overview:
    Input: (batch_size, channels, H, W), timestep
    
    Encoder (Downsampling path)

    Middle
    
    Decoder (Upsampling path)
    
    Output: (batch_size, channels, H, W)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .blocks import (
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


class UNet(nn.Module):
    """
    TODO: design your own U-Net architecture for diffusion models.

    Args:
        in_channels: Number of input image channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel count (multiplied by channel_mult at each level)
        channel_mult: Tuple of channel multipliers for each resolution level
                     e.g., (1, 2, 4, 8) means channels are [C, 2C, 4C, 8C]
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply self-attention
                              e.g., [16, 8] applies attention at 16x16 and 8x8
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use FiLM conditioning in ResBlocks
        condition_num_classes: Optional number of classes for each discrete
                              semantic condition field.
    
    Example:
        >>> model = UNet(
        ...     in_channels=3,
        ...     out_channels=3, 
        ...     base_channels=128,
        ...     channel_mult=(1, 2, 2, 4),
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (4,))
        >>> out = model(x, t)
        >>> out.shape
        torch.Size([4, 3, 64, 64])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
        condition_num_classes: Optional[List[int]] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        self.condition_num_classes = condition_num_classes

        time_embed_dim = base_channels * 4
        self.time_embedding = TimestepEmbedding(time_embed_dim)
        self.condition_embeddings = None
        if condition_num_classes is not None:
            self.condition_embeddings = nn.ModuleList(
                nn.Embedding(num_classes, time_embed_dim)
                for num_classes in condition_num_classes
            )

        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.input_blocks = nn.ModuleList()
        self.output_blocks = nn.ModuleList()
        input_block_channels = [base_channels]

        channels = base_channels
        resolution = 64
        for level, mult in enumerate(channel_mult):
            out_channels_level = base_channels * mult
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        channels,
                        out_channels_level,
                        time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                channels = out_channels_level
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(channels, num_heads=num_heads))
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_channels.append(channels)

            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.ModuleList([Downsample(channels)]))
                input_block_channels.append(channels)
                resolution //= 2

        self.middle_block = nn.ModuleList([
            ResBlock(
                channels,
                channels,
                time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(channels, num_heads=num_heads),
            ResBlock(
                channels,
                channels,
                time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        ])

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_channels_level = base_channels * mult
            for block_idx in range(num_res_blocks + 1):
                skip_channels = input_block_channels.pop()
                layers = [
                    ResBlock(
                        channels + skip_channels,
                        out_channels_level,
                        time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                channels = out_channels_level
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(channels, num_heads=num_heads))
                if level != 0 and block_idx == num_res_blocks:
                    layers.append(Upsample(channels))
                    resolution *= 2
                self.output_blocks.append(nn.ModuleList(layers))

        self.out = nn.Sequential(
            GroupNorm32(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        TODO: Implement the forward pass of the unet
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
               This is typically the noisy image x_t
            t: Timestep tensor of shape (batch_size,)
            condition: Optional long tensor of shape (batch_size, num_fields)
                with discrete semantic control ids.

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """

        time_emb = self.time_embedding(t)
        if self.condition_embeddings is not None:
            if condition is None:
                condition = torch.zeros(
                    x.shape[0],
                    len(self.condition_embeddings),
                    dtype=torch.long,
                    device=x.device,
                )
            if condition.ndim != 2 or condition.shape[1] != len(self.condition_embeddings):
                raise ValueError(
                    "condition must have shape "
                    f"(batch_size, {len(self.condition_embeddings)}), got {tuple(condition.shape)}."
                )
            condition = condition.to(device=x.device, dtype=torch.long)
            condition_emb = torch.zeros_like(time_emb)
            for field_idx, embedding in enumerate(self.condition_embeddings):
                condition_emb = condition_emb + embedding(condition[:, field_idx])
            time_emb = time_emb + condition_emb

        h = self.input_conv(x)
        skips = [h]

        for block in self.input_blocks:
            h = self._forward_block(block, h, time_emb)
            skips.append(h)

        h = self._forward_block(self.middle_block, h, time_emb)

        for block in self.output_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = self._forward_block(block, h, time_emb)

        return self.out(h)

    @staticmethod
    def _forward_block(block: nn.ModuleList, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Run a mixed block of time-conditioned and plain layers."""
        h = x
        for layer in block:
            if isinstance(layer, ResBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)
        return h


def create_model_from_config(config: dict) -> UNet:
    """
    Factory function to create a UNet from a configuration dictionary.
    
    Args:
        config: Dictionary containing model configuration
                Expected to have a 'model' key with the relevant parameters
    
    Returns:
        Instantiated UNet model
    """
    model_config = config['model']
    data_config = config['data']
    
    return UNet(
        in_channels=data_config['channels'],
        out_channels=data_config['channels'],
        base_channels=model_config['base_channels'],
        channel_mult=tuple(model_config['channel_mult']),
        num_res_blocks=model_config['num_res_blocks'],
        attention_resolutions=model_config['attention_resolutions'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        use_scale_shift_norm=model_config['use_scale_shift_norm'],
        condition_num_classes=model_config.get('condition_num_classes'),
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    print("Testing UNet...")
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.rand(batch_size)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful!")
