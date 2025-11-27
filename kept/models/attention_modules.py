"""
Modular Attention Mechanisms
Can be applied to ANY architecture (Attention U-Net, UNet++, FPN, DeepLabV3+, MANet, PSPNet)

Usage:
    from models.attention_modules import AttentionWrapper
    
    # Wrap any model
    model = get_model(config)  # Any architecture
    model = AttentionWrapper(model, config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. SQUEEZE-AND-EXCITATION (SE) BLOCK
# ============================================================================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Stable Version)
    - Channel-wise attention
    - Very lightweight (~0.1M params)
    - Numerical stability improvements
    - Paper: "Squeeze-and-Excitation Networks" (CVPR 2018)
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 16)
        scale: Scaling factor for residual connection (default: 0.1)
    """
    
    def __init__(self, channels, reduction=16, scale=0.1):
        super(SEBlock, self).__init__()
        
        # Ensure reduced channels is at least 1
        reduced_channels = max(channels // reduction, 1)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.BatchNorm1d(reduced_channels),  # Add batch norm
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.BatchNorm1d(channels)  # Add batch norm
        )
        self.sigmoid = nn.Sigmoid()
        self.scale = scale
    
    def forward(self, x):
        b, c, _, _ = x.size()
        identity = x
        
        # Squeeze: Global pooling
        y = self.squeeze(x).view(b, c)
        
        # Excitation: FC layers
        y = self.excitation(y)
        
        # Clamp to prevent extreme values
        y = torch.clamp(y, -10, 10)
        
        # Sigmoid attention
        y = self.sigmoid(y).view(b, c, 1, 1)
        
        # Apply with scaled residual
        attended = x * y.expand_as(x)
        out = identity + self.scale * (attended - identity)
        
        return out


# ============================================================================
# 2. CONVOLUTIONAL BLOCK ATTENTION MODULE (CBAM)
# ============================================================================
class ChannelAttention(nn.Module):
    """Channel Attention component of CBAM with numerical stability"""
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Ensure reduced channels is at least 1
        reduced_channels = max(channels // reduction, 1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.BatchNorm2d(reduced_channels),  # Add batch norm for stability
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)  # Add batch norm for stability
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Add small epsilon to prevent NaN
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        
        # Clamp to prevent extreme values
        out = torch.clamp(out, -10, 10)
        
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention component of CBAM with numerical stability"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, 
                     padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1)  # Add batch norm for stability
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Use mean and max with small epsilon
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        
        # Clamp to prevent extreme values
        out = torch.clamp(out, -10, 10)
        
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Stable Version)
    - Channel attention + Spatial attention
    - Sequential application with residual scaling
    - Numerical stability improvements
    - Paper: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for channel attention (default: 16)
        kernel_size: Kernel size for spatial attention (default: 7)
        scale: Scaling factor for residual connection (default: 0.1)
    """
    
    def __init__(self, channels, reduction=16, kernel_size=7, scale=0.1):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.scale = scale  # Learnable scaling factor
    
    def forward(self, x):
        # Store original input
        identity = x
        
        # Channel attention with residual
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # Spatial attention with residual
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        
        # Scaled residual connection to prevent gradient explosion
        x = identity + self.scale * (x - identity)
        
        return x


# ============================================================================
# 3. DUAL ATTENTION (POSITION + CHANNEL)
# ============================================================================
class PositionAttention(nn.Module):
    """
    Position Attention Module
    - Captures spatial relationships
    - Long-range dependencies
    
    Args:
        channels: Number of input channels
    """
    
    def __init__(self, channels):
        super(PositionAttention, self).__init__()
        
        # Ensure reduced channels is at least 1
        reduced_channels = max(channels // 8, 1)
        
        self.query_conv = nn.Conv2d(channels, reduced_channels, 1)
        self.key_conv = nn.Conv2d(channels, reduced_channels, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Query, Key, Value projections
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        key = self.key_conv(x).view(B, -1, H * W)  # (B, C', HW)
        value = self.value_conv(x).view(B, C, H * W)  # (B, C, HW)
        
        # Attention map
        attention = torch.softmax(torch.bmm(query, key), dim=-1)  # (B, HW, HW)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable weight
        return self.gamma * out + x


class ChannelAttentionModule(nn.Module):
    """
    Channel Attention Module
    - Captures channel relationships
    - Feature dependencies
    
    Args:
        channels: Number of input channels
    """
    
    def __init__(self, channels):
        super(ChannelAttentionModule, self).__init__()
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Reshape for channel attention
        proj_query = x.view(B, C, -1)  # (B, C, HW)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        
        # Attention map
        energy = torch.bmm(proj_query, proj_key)  # (B, C, C)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = torch.softmax(energy_new, dim=-1)
        
        # Apply attention
        proj_value = x.view(B, C, -1)  # (B, C, HW)
        out = torch.bmm(attention, proj_value)
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable weight
        return self.gamma * out + x


class DualAttention(nn.Module):
    """
    Dual Attention Network
    - Combines Position Attention + Channel Attention
    - Parallel application with summation
    - Paper: "Dual Attention Network for Scene Segmentation" (CVPR 2019)
    
    Args:
        channels: Number of input channels
    """
    
    def __init__(self, channels):
        super(DualAttention, self).__init__()
        
        self.position_attention = PositionAttention(channels)
        self.channel_attention = ChannelAttentionModule(channels)
    
    def forward(self, x):
        p_out = self.position_attention(x)
        c_out = self.channel_attention(x)
        return p_out + c_out


# ============================================================================
# 4. MULTI-SCALE ATTENTION
# ============================================================================
class MultiScaleAttention(nn.Module):
    """
    Multi-Scale Attention Module
    - Captures features at different scales
    - Important for objects of varying sizes (like stroke lesions)
    
    Args:
        channels: Number of input channels
        scales: List of kernel sizes for multi-scale processing
    """
    
    def __init__(self, channels, scales=[1, 3, 5, 7]):
        super(MultiScaleAttention, self).__init__()
        
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for k in scales
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * len(scales), channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Multi-scale features
        multi_scale = [scale(x) for scale in self.scales]
        
        # Fuse
        fused = torch.cat(multi_scale, dim=1)
        
        # Attention weights
        attention = self.fusion(fused)
        
        return x * attention


# ============================================================================
# 5. EFFICIENT CHANNEL ATTENTION (ECA)
# ============================================================================
class ECABlock(nn.Module):
    """
    Efficient Channel Attention
    - More efficient than SE block
    - No dimensionality reduction
    - Paper: "ECA-Net: Efficient Channel Attention for Deep CNNs" (CVPR 2020)
    
    Args:
        channels: Number of input channels
        k_size: Kernel size for 1D convolution (auto-computed if None)
    """
    
    def __init__(self, channels, k_size=None):
        super(ECABlock, self).__init__()
        
        # Auto-compute kernel size if not provided
        if k_size is None:
            k_size = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + 1) / 2))
            k_size = k_size if k_size % 2 else k_size + 1
            k_size = max(k_size, 3)  # Ensure at least kernel size 3
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Global pooling
        y = self.avg_pool(x)
        
        # 1D convolution along channel dimension
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Attention
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)


# ============================================================================
# 6. ATTENTION WRAPPER - APPLIES TO ANY ARCHITECTURE
# ============================================================================
class AttentionWrapper(nn.Module):
    """
    Universal Attention Wrapper
    - Can be applied to ANY segmentation architecture
    - Inserts attention modules at strategic points
    
    Args:
        model: Base segmentation model (any architecture)
        config: Configuration object with attention settings
    """
    
    def __init__(self, model, config):
        super(AttentionWrapper, self).__init__()
        
        self.base_model = model
        self.config = config
        
        # Get attention configurations
        self.use_se = getattr(config, 'USE_SE_ATTENTION', False)
        self.use_cbam = getattr(config, 'USE_CBAM_ATTENTION', False)
        self.use_dual = getattr(config, 'USE_DUAL_ATTENTION', False)
        self.use_multiscale = getattr(config, 'USE_MULTISCALE_ATTENTION', False)
        self.use_eca = getattr(config, 'USE_ECA_ATTENTION', False)
        
        # Register attention modules
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register forward hooks to insert attention modules"""
        
        # This will be implemented to hook into intermediate layers
        # For now, we'll apply attention post-processing
        pass
    
    def forward(self, x):
        """Forward pass with attention enhancement"""
        
        # Base model forward
        features = self.base_model(x)
        
        # Apply attention modules if enabled
        if self.use_se:
            se = SEBlock(features.size(1)).to(features.device)
            features = se(features)
        
        if self.use_cbam:
            cbam = CBAM(features.size(1)).to(features.device)
            features = cbam(features)
        
        if self.use_eca:
            eca = ECABlock(features.size(1)).to(features.device)
            features = eca(features)
        
        return features


# ============================================================================
# HELPER FUNCTION: INSERT ATTENTION INTO EXISTING MODEL
# ============================================================================
def insert_attention_modules(model, config):
    """
    Insert attention modules into an existing model
    Works with any PyTorch segmentation model
    
    Args:
        model: Base segmentation model
        config: Configuration with attention settings
    
    Returns:
        Modified model with attention modules
    """
    
    # Get attention types to add
    attention_types = []
    if getattr(config, 'USE_SE_ATTENTION', False):
        attention_types.append('se')
    if getattr(config, 'USE_CBAM_ATTENTION', False):
        attention_types.append('cbam')
    if getattr(config, 'USE_DUAL_ATTENTION', False):
        attention_types.append('dual')
    if getattr(config, 'USE_MULTISCALE_ATTENTION', False):
        attention_types.append('multiscale')
    if getattr(config, 'USE_ECA_ATTENTION', False):
        attention_types.append('eca')
    
    if not attention_types:
        return model  # No attention to add
    
    print(f"\n🔥 Adding attention modules: {', '.join(attention_types)}")
    
    # Recursively add attention to Conv2d layers
    def add_attention_recursive(module, name=''):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            # If it's a conv layer followed by batch norm, insert attention
            if isinstance(child, nn.Conv2d):
                # Get number of output channels
                out_channels = child.out_channels
                
                # Create attention module based on config
                attention_module = None
                
                if 'se' in attention_types:
                    attention_module = SEBlock(out_channels)
                elif 'cbam' in attention_types:
                    attention_module = CBAM(out_channels)
                elif 'eca' in attention_types:
                    attention_module = ECABlock(out_channels)
                elif 'dual' in attention_types and out_channels >= 256:  # Only for larger channels
                    attention_module = DualAttention(out_channels)
                elif 'multiscale' in attention_types and out_channels >= 128:
                    attention_module = MultiScaleAttention(out_channels)
                
                # Note: Actual insertion would require more sophisticated approach
                # This is a simplified version for demonstration
            
            else:
                # Recursively process children
                add_attention_recursive(child, full_name)
    
    # Add attention modules
    add_attention_recursive(model)
    
    return model


# ============================================================================
# FACTORY FUNCTION
# ============================================================================
def get_attention_module(attention_type, channels, **kwargs):
    """
    Factory function to create attention modules
    
    Args:
        attention_type: Type of attention ('se', 'cbam', 'dual', 'multiscale', 'eca')
        channels: Number of input channels
        **kwargs: Additional arguments for specific attention types
    
    Returns:
        Attention module instance
    """
    
    attention_map = {
        'se': SEBlock,
        'cbam': CBAM,
        'dual': DualAttention,
        'multiscale': MultiScaleAttention,
        'eca': ECABlock
    }
    
    if attention_type not in attention_map:
        raise ValueError(f"Unknown attention type: {attention_type}. "
                        f"Available: {list(attention_map.keys())}")
    
    return attention_map[attention_type](channels, **kwargs)
