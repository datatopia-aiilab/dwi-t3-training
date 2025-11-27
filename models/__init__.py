"""
Model Factory for Multiple Architecture Support
Provides unified interface for all segmentation models
"""

import torch
import torch.nn as nn


def get_model(config):
    """
    Factory function to get the specified model architecture
    
    Args:
        config: Configuration module with MODEL_ARCHITECTURE, IN_CHANNELS, OUT_CHANNELS, etc.
    
    Returns:
        PyTorch model instance
    
    Supported Architectures:
        - attention_unet: Custom Attention U-Net (baseline)
        - attention_unet_ds: Attention U-Net with Deep Supervision ⭐ NEW
        - unet++: U-Net++ with nested skip connections
        - fpn: Feature Pyramid Network
        - deeplabv3+: DeepLabV3+ with ASPP
        - manet: Multi-Attention Network
        - pspnet: Pyramid Scene Parsing Network
    """
    
    arch = config.MODEL_ARCHITECTURE.lower()
    
    print(f"\n{'='*70}")
    print(f"🏗️  Loading Model Architecture: {arch.upper()}")
    print(f"{'='*70}")
    
    if arch == 'attention_unet':
        from .attention_unet import AttentionUNet
        model = AttentionUNet(config)
        print(f"✅ Loaded Custom Attention U-Net")
        print(f"   Encoder channels: {config.ENCODER_CHANNELS}")
        print(f"   Bottleneck: {config.BOTTLENECK_CHANNELS}")
        print(f"   Attention gates: {config.USE_ATTENTION}")
    
    elif arch == 'attention_unet_ds' or arch == 'attention_unet_deepsupervision':
        from .attention_unet import AttentionUNetDeepSupervision
        model = AttentionUNetDeepSupervision(
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            encoder_channels=config.ENCODER_CHANNELS,
            bottleneck_channels=config.BOTTLENECK_CHANNELS,
            use_attention=config.USE_ATTENTION,
            dropout=config.DROPOUT,
            use_se=getattr(config, 'USE_SE_ATTENTION', False),
            use_cbam=getattr(config, 'USE_CBAM_ATTENTION', False),
            use_eca=getattr(config, 'USE_ECA_ATTENTION', False),
            use_dual=getattr(config, 'USE_DUAL_ATTENTION', False),
            use_multiscale=getattr(config, 'USE_MULTISCALE_ATTENTION', False),
            num_supervision_levels=getattr(config, 'DEEP_SUPERVISION_LEVELS', 3)
        )
        print(f"✅ Loaded Custom Attention U-Net with Deep Supervision")
        print(f"   Encoder channels: {config.ENCODER_CHANNELS}")
        print(f"   Bottleneck: {config.BOTTLENECK_CHANNELS}")
        print(f"   Attention gates: {config.USE_ATTENTION}")
        print(f"   🔥 Deep Supervision Levels: {getattr(config, 'DEEP_SUPERVISION_LEVELS', 3)}")
        
    elif arch == 'unet++' or arch == 'unetplusplus':
        from .smp_wrapper import UNetPlusPlusWrapper
        model = UNetPlusPlusWrapper(config)
        print(f"✅ Loaded U-Net++")
        print(f"   Encoder: {config.ENCODER_NAME}")
        print(f"   Pre-trained: {config.ENCODER_WEIGHTS}")
        
    elif arch == 'fpn':
        from .smp_wrapper import FPNWrapper
        model = FPNWrapper(config)
        print(f"✅ Loaded FPN (Feature Pyramid Network)")
        print(f"   Encoder: {config.ENCODER_NAME}")
        print(f"   Pre-trained: {config.ENCODER_WEIGHTS}")
        
    elif arch == 'deeplabv3+' or arch == 'deeplabv3plus':
        from .smp_wrapper import DeepLabV3PlusWrapper
        model = DeepLabV3PlusWrapper(config)
        print(f"✅ Loaded DeepLabV3+")
        print(f"   Encoder: {config.ENCODER_NAME}")
        print(f"   Pre-trained: {config.ENCODER_WEIGHTS}")
        
    elif arch == 'manet':
        from .smp_wrapper import MANetWrapper
        model = MANetWrapper(config)
        print(f"✅ Loaded MANet (Multi-Attention Network)")
        print(f"   Encoder: {config.ENCODER_NAME}")
        print(f"   Pre-trained: {config.ENCODER_WEIGHTS}")
        
    elif arch == 'pspnet':
        from .smp_wrapper import PSPNetWrapper
        model = PSPNetWrapper(config)
        print(f"✅ Loaded PSPNet (Pyramid Scene Parsing)")
        print(f"   Encoder: {config.ENCODER_NAME}")
        print(f"   Pre-trained: {config.ENCODER_WEIGHTS}")
        
    else:
        raise ValueError(
            f"Unknown architecture: {arch}\n"
            f"Available options: attention_unet, attention_unet_ds, unet++, fpn, deeplabv3+, manet, pspnet"
        )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"   Input channels: {config.IN_CHANNELS}")
    print(f"   Output channels: {config.OUT_CHANNELS}")
    print(f"{'='*70}\n")
    
    return model


def get_attention_unet(config):
    """
    Backward compatibility wrapper for old code
    """
    print("⚠️  Using deprecated function 'get_attention_unet()'")
    print("   Please update to 'get_model()' for multi-architecture support")
    
    # Force attention_unet architecture
    original_arch = getattr(config, 'MODEL_ARCHITECTURE', 'attention_unet')
    config.MODEL_ARCHITECTURE = 'attention_unet'
    
    model = get_model(config)
    
    # Restore original
    config.MODEL_ARCHITECTURE = original_arch
    
    return model


# Export main function
__all__ = ['get_model', 'get_attention_unet']
