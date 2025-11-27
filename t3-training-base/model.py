"""
Simple Attention U-Net Model
Standard U-Net with Attention Gates - No complexity!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard Conv Block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class AttentionGate(nn.Module):
    """Attention Gate for focusing on relevant features"""
    
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from coarser scale (decoder)
            x: Input features from encoder (skip connection)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """
    Simple Attention U-Net
    
    Architecture:
        Encoder: 4 levels (64, 128, 256, 512)
        Bottleneck: 1024
        Decoder: 4 levels with Attention Gates
        Output: 1 channel (binary segmentation)
    """
    
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super(AttentionUNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        
        # Decoder with Attention Gates
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=base_channels * 8, F_l=base_channels * 8, F_int=base_channels * 4)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=base_channels * 4, F_l=base_channels * 4, F_int=base_channels * 2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=base_channels * 2, F_l=base_channels * 2, F_int=base_channels)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=base_channels, F_l=base_channels, F_int=base_channels // 2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        
        # Output
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder with Attention
        d4 = self.up4(b)
        e4_att = self.att4(g=d4, x=e4)
        d4 = torch.cat([e4_att, d4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = torch.cat([e3_att, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = torch.cat([e2_att, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = torch.cat([e1_att, d1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        out = self.out(d1)
        return torch.sigmoid(out)


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Test model
    model = AttentionUNet(in_channels=3, out_channels=1, base_channels=64)
    
    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Attention U-Net")
    print(f"{'='*60}")
    print(f"Total parameters: {total:,} ({total/1e6:.1f}M)")
    print(f"Trainable parameters: {trainable:,} ({trainable/1e6:.1f}M)")
    print(f"{'='*60}\n")
    
    # Test forward pass
    x = torch.randn(2, 3, 384, 384)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"\n✅ Model test passed!")
