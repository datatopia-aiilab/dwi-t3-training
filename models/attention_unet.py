"""
Attention U-Net Model for DWI Ischemic Stroke Segmentation
Implements 2.5D Attention U-Net with skip connections and attention gates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutional Block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        if self.dropout:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


class AttentionGate(nn.Module):
    """
    Attention Gate module for Attention U-Net
    Helps the model focus on relevant regions
    
    Args:
        in_channels: Number of channels in input feature map (from encoder)
        gating_channels: Number of channels in gating signal (from decoder)
        inter_channels: Number of intermediate channels (default: in_channels // 2)
    """
    
    def __init__(self, in_channels, gating_channels, inter_channels=None):
        super(AttentionGate, self).__init__()
        
        if inter_channels is None:
            inter_channels = in_channels // 2
        
        # Transform input features
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Output transformation
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, g):
        """
        Args:
            x: Input feature map from encoder (skip connection)
            g: Gating signal from decoder (coarser scale)
        
        Returns:
            Attention-weighted feature map
        """
        # Transform input
        x_transformed = self.W_x(x)
        g_transformed = self.W_g(g)
        
        # If g is smaller than x, upsample it
        if g_transformed.shape[2:] != x_transformed.shape[2:]:
            g_transformed = F.interpolate(g_transformed, size=x_transformed.shape[2:], 
                                         mode='bilinear', align_corners=True)
        
        # Add and activate
        combined = self.relu(x_transformed + g_transformed)
        
        # Generate attention coefficients
        attention = self.psi(combined)
        
        # Apply attention
        return x * attention


class EncoderBlock(nn.Module):
    """Encoder block: ConvBlock + MaxPooling"""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(EncoderBlock, self).__init__()
        
        self.conv = ConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        """Returns: x_conv (for skip), x_pool (for next encoder)"""
        x_conv = self.conv(x)
        x_pool = self.pool(x_conv)
        return x_conv, x_pool


class DecoderBlock(nn.Module):
    """Decoder block: Upsampling + Attention Gate + Concatenation + ConvBlock"""
    
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True, dropout=0.0):
        super(DecoderBlock, self).__init__()
        
        self.use_attention = use_attention
        
        # Upsampling
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Attention gate
        if use_attention:
            self.attention = AttentionGate(skip_channels, in_channels // 2)
        
        # Convolutional block
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels, dropout)
    
    def forward(self, x, skip):
        """
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from encoder
        """
        # Upsample
        x = self.up(x)
        
        # Apply attention to skip connection
        if self.use_attention:
            skip = self.attention(skip, x)
        
        # Concatenate
        x = torch.cat([x, skip], dim=1)
        
        # Apply convolutions
        x = self.conv(x)
        
        return x


class AttentionUNet(nn.Module):
    """
    Attention U-Net for 2.5D Medical Image Segmentation
    
    Input: (B, 3, H, W) - 2.5D input [N-1, N, N+1] slices
    Output: (B, 1, H, W) - Binary segmentation mask [0, 1]
    
    Args:
        config: Configuration object with model parameters
    """
    
    def __init__(self, config):
        super(AttentionUNet, self).__init__()
        
        in_channels = config.IN_CHANNELS
        out_channels = config.OUT_CHANNELS
        encoder_channels = config.ENCODER_CHANNELS
        bottleneck_channels = config.BOTTLENECK_CHANNELS
        use_attention = config.USE_ATTENTION
        dropout = 0.0
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        
        # ==================== Encoder ====================
        self.encoders = nn.ModuleList()
        
        prev_channels = in_channels
        for channels in encoder_channels:
            self.encoders.append(EncoderBlock(prev_channels, channels, dropout))
            prev_channels = channels
        
        # ==================== Bottleneck ====================
        self.bottleneck = ConvBlock(encoder_channels[-1], bottleneck_channels, dropout)
        
        # ==================== Decoder ====================
        self.decoders = nn.ModuleList()
        
        decoder_channels = encoder_channels[::-1]  # Reverse order
        
        prev_channels = bottleneck_channels
        for i, channels in enumerate(decoder_channels):
            skip_channels = channels
            out_ch = channels if i < len(decoder_channels) - 1 else decoder_channels[-1]
            
            self.decoders.append(
                DecoderBlock(prev_channels, skip_channels, out_ch, use_attention, dropout)
            )
            prev_channels = out_ch
        
        # ==================== Output ====================
        self.output = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, H, W) for 2.5D input
        
        Returns:
            Output tensor (B, 1, H, W) with segmentation probabilities
        """
        # Encoder Path
        skip_connections = []
        
        for encoder in self.encoders:
            x_skip, x = encoder(x)
            skip_connections.append(x_skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder Path
        skip_connections = skip_connections[::-1]
        
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
        
        # Output
        x = self.output(x)
        
        return x
