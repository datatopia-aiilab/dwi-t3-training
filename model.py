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
    
    This is the CORE innovation that helps the model focus on relevant regions
    
    Paper: "Attention U-Net: Learning Where to Look for the Pancreas" (Oktay et al., 2018)
    
    How it works:
        1. Takes gating signal (g) from decoder and feature map (x) from encoder
        2. Processes both through 1x1 convolutions
        3. Adds them together and applies ReLU
        4. Passes through another 1x1 conv and sigmoid to get attention coefficients
        5. Multiplies attention coefficients with input features
    
    This allows the model to suppress irrelevant regions and highlight salient features
    
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
        
        # Transform gating signal (may need upsampling if sizes don't match)
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
    """
    Encoder block: ConvBlock + MaxPooling
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dropout: Dropout probability
    """
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(EncoderBlock, self).__init__()
        
        self.conv = ConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        """
        Returns:
            x_conv: Feature map before pooling (for skip connection)
            x_pool: Pooled feature map (for next encoder block)
        """
        x_conv = self.conv(x)
        x_pool = self.pool(x_conv)
        
        return x_conv, x_pool


class DecoderBlock(nn.Module):
    """
    Decoder block: Upsampling + Attention Gate + Concatenation + ConvBlock
    
    Args:
        in_channels: Number of input channels (from previous decoder)
        skip_channels: Number of channels in skip connection
        out_channels: Number of output channels
        use_attention: Whether to use attention gate
        dropout: Dropout probability
    """
    
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
        
        Returns:
            Output feature map
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
    
    Architecture:
        - Encoder: 4 levels with increasing channels [64, 128, 256, 512]
        - Bottleneck: 1024 channels
        - Decoder: 4 levels with attention gates and skip connections
        - Output: 1 channel with sigmoid activation (binary segmentation)
    
    Input: (B, 3, H, W) - 2.5D input [N-1, N, N+1] slices
    Output: (B, 1, H, W) - Binary segmentation mask [0, 1]
    
    Args:
        in_channels: Number of input channels (default: 3 for 2.5D)
        out_channels: Number of output channels (default: 1 for binary)
        encoder_channels: List of encoder channel sizes
        bottleneck_channels: Number of channels in bottleneck
        use_attention: Whether to use attention gates
        dropout: Dropout probability
    """
    
    def __init__(self, 
                 in_channels=3, 
                 out_channels=1,
                 encoder_channels=[64, 128, 256, 512],
                 bottleneck_channels=1024,
                 use_attention=True,
                 dropout=0.0):
        super(AttentionUNet, self).__init__()
        
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
            skip_channels = channels  # Channels from corresponding encoder
            out_ch = channels if i < len(decoder_channels) - 1 else decoder_channels[-1]
            
            self.decoders.append(
                DecoderBlock(prev_channels, skip_channels, out_ch, use_attention, dropout)
            )
            prev_channels = out_ch
        
        # ==================== Output ====================
        self.output = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1),
            nn.Sigmoid()  # Output probabilities [0, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, H, W) for 2.5D input
        
        Returns:
            Output tensor (B, 1, H, W) with segmentation probabilities
        """
        # ==================== Encoder Path ====================
        skip_connections = []
        
        for encoder in self.encoders:
            x_skip, x = encoder(x)
            skip_connections.append(x_skip)
        
        # ==================== Bottleneck ====================
        x = self.bottleneck(x)
        
        # ==================== Decoder Path ====================
        skip_connections = skip_connections[::-1]  # Reverse order
        
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
        
        # ==================== Output ====================
        x = self.output(x)
        
        return x
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params
        }


# ==================== Model Factory ====================

def get_attention_unet(config=None):
    """
    Factory function to create Attention U-Net from config
    
    Args:
        config: Configuration module or dict with model parameters
    
    Returns:
        model: AttentionUNet instance
    """
    if config is None:
        # Default parameters
        return AttentionUNet()
    
    # Extract parameters from config
    if hasattr(config, 'IN_CHANNELS'):
        # It's a module
        in_channels = config.IN_CHANNELS
        out_channels = config.OUT_CHANNELS
        encoder_channels = config.ENCODER_CHANNELS
        bottleneck_channels = config.BOTTLENECK_CHANNELS
        use_attention = config.USE_ATTENTION
    else:
        # It's a dict
        in_channels = config.get('in_channels', 3)
        out_channels = config.get('out_channels', 1)
        encoder_channels = config.get('encoder_channels', [64, 128, 256, 512])
        bottleneck_channels = config.get('bottleneck_channels', 1024)
        use_attention = config.get('use_attention', True)
    
    model = AttentionUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        encoder_channels=encoder_channels,
        bottleneck_channels=bottleneck_channels,
        use_attention=use_attention
    )
    
    return model


# ==================== Testing Functions ====================

def test_model():
    """ทดสอบ model architecture"""
    print("🧪 Testing Attention U-Net Model...\n")
    
    # Test 1: Model creation
    print("Test 1: Model Creation")
    print("="*60)
    model = AttentionUNet(
        in_channels=3,
        out_channels=1,
        encoder_channels=[64, 128, 256, 512],
        bottleneck_channels=1024,
        use_attention=True
    )
    
    params = model.count_parameters()
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Model size: ~{params['total'] * 4 / (1024**2):.2f} MB (FP32)")
    
    # Test 2: Forward pass
    print("\n" + "="*60)
    print("Test 2: Forward Pass")
    print("="*60)
    
    batch_sizes = [1, 2, 4]
    input_sizes = [(256, 256), (128, 128), (512, 512)]
    
    for batch_size in batch_sizes:
        for h, w in input_sizes:
            x = torch.randn(batch_size, 3, h, w)
            
            # Forward pass
            with torch.no_grad():
                y = model(x)
            
            print(f"Input: {x.shape} -> Output: {y.shape}")
            print(f"  Output range: [{y.min():.3f}, {y.max():.3f}]")
            
            # Check output shape
            assert y.shape == (batch_size, 1, h, w), f"Output shape mismatch!"
            assert y.min() >= 0 and y.max() <= 1, f"Output not in [0, 1] range!"
    
    # Test 3: Gradient flow
    print("\n" + "="*60)
    print("Test 3: Gradient Flow")
    print("="*60)
    
    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 128, 128)).float()
    
    y = model(x)
    loss = F.binary_cross_entropy(y, target)
    loss.backward()
    
    print(f"Loss: {loss.item():.6f}")
    print(f"Input gradient: {x.grad is not None}")
    print(f"Input gradient mean: {x.grad.mean().item():.6f}")
    
    # Check all parameters have gradients
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"Parameters with gradients: {params_with_grad}/{total_params}")
    
    # Test 4: Model with and without attention
    print("\n" + "="*60)
    print("Test 4: Attention vs No Attention")
    print("="*60)
    
    model_with_att = AttentionUNet(use_attention=True)
    model_without_att = AttentionUNet(use_attention=False)
    
    params_with = model_with_att.count_parameters()['total']
    params_without = model_without_att.count_parameters()['total']
    
    print(f"With Attention: {params_with:,} parameters")
    print(f"Without Attention: {params_without:,} parameters")
    print(f"Difference: {params_with - params_without:,} parameters")
    print(f"Attention overhead: {(params_with - params_without) / params_without * 100:.2f}%")
    
    # Test 5: Memory usage estimation
    print("\n" + "="*60)
    print("Test 5: Memory Usage Estimation")
    print("="*60)
    
    batch_size = 8
    input_size = (256, 256)
    
    # Input
    input_mem = batch_size * 3 * input_size[0] * input_size[1] * 4 / (1024**2)
    
    # Model weights
    model_mem = params['total'] * 4 / (1024**2)
    
    # Activations (rough estimate - actual may vary)
    # Encoder: ~8x input size (with all feature maps)
    # Decoder: ~4x input size
    activation_mem = input_mem * 12
    
    # Gradients (same size as weights)
    grad_mem = model_mem
    
    # Optimizer state (Adam: 2x weights for momentum and variance)
    optimizer_mem = model_mem * 2
    
    total_mem = input_mem + model_mem + activation_mem + grad_mem + optimizer_mem
    
    print(f"Estimated memory usage (batch_size={batch_size}, input={input_size}):")
    print(f"  Input: {input_mem:.2f} MB")
    print(f"  Model weights: {model_mem:.2f} MB")
    print(f"  Activations: {activation_mem:.2f} MB")
    print(f"  Gradients: {grad_mem:.2f} MB")
    print(f"  Optimizer state: {optimizer_mem:.2f} MB")
    print(f"  Total: {total_mem:.2f} MB")
    print(f"\n  Recommended GPU VRAM: >= {total_mem * 1.5:.0f} MB (~{total_mem * 1.5 / 1024:.1f} GB)")
    
    print("\n✅ All model tests passed!")


if __name__ == "__main__":
    test_model()
