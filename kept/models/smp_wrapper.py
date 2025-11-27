"""
Segmentation Models PyTorch (SMP) Wrappers
Provides unified interface for all SMP-based architectures with 2.5D support
+ Optional additional attention mechanisms
"""

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("⚠️  WARNING: segmentation_models_pytorch not installed!")
    print("   Install with: pip install segmentation-models-pytorch")

# Import attention modules
try:
    from .attention_modules import SEBlock, CBAM, ECABlock, DualAttention, MultiScaleAttention
    ATTENTION_AVAILABLE = True
except ImportError:
    ATTENTION_AVAILABLE = False
    SEBlock = CBAM = ECABlock = DualAttention = MultiScaleAttention = None


class AttentionInjector(nn.Module):
    """
    Wrapper to inject attention modules into any model
    Works by wrapping the model and applying attention to output features
    """
    
    def __init__(self, base_model, config):
        super().__init__()
        
        self.base_model = base_model
        
        # Get attention configs
        use_se = getattr(config, 'USE_SE_ATTENTION', False)
        use_cbam = getattr(config, 'USE_CBAM_ATTENTION', False)
        use_eca = getattr(config, 'USE_ECA_ATTENTION', False)
        
        # Store flags
        self.use_se = use_se and ATTENTION_AVAILABLE
        self.use_cbam = use_cbam and ATTENTION_AVAILABLE
        self.use_eca = use_eca and ATTENTION_AVAILABLE
        
        # Will be initialized on first forward pass
        self._attention_initialized = False
        
        # Placeholder for attention modules (registered as None initially)
        self.se_module = None
        self.cbam_module = None
        self.eca_module = None
        
        # Print summary
        if any([self.use_se, self.use_cbam, self.use_eca]):
            attention_types = []
            if self.use_se:
                attention_types.append("SE")
            if self.use_cbam:
                attention_types.append("CBAM")
            if self.use_eca:
                attention_types.append("ECA")
            print(f"   🔥 Injecting attention: {', '.join(attention_types)}")
    
    def _initialize_attention(self, channels):
        """Initialize attention modules with correct number of channels"""
        if self.use_se:
            self.se_module = SEBlock(channels)
        if self.use_cbam:
            self.cbam_module = CBAM(channels)
        if self.use_eca:
            self.eca_module = ECABlock(channels)
        
        self._attention_initialized = True
    
    def forward(self, x):
        # Base model forward
        out = self.base_model(x)
        
        # Initialize attention modules on first forward pass
        if not self._attention_initialized:
            channels = out.size(1)
            self._initialize_attention(channels)
            
            # Move to same device as output
            if self.se_module is not None:
                self.se_module = self.se_module.to(out.device)
            if self.cbam_module is not None:
                self.cbam_module = self.cbam_module.to(out.device)
            if self.eca_module is not None:
                self.eca_module = self.eca_module.to(out.device)
        
        # Apply attention modules
        if self.se_module is not None:
            out = self.se_module(out)
        if self.cbam_module is not None:
            out = self.cbam_module(out)
        if self.eca_module is not None:
            out = self.eca_module(out)
        
        return out


class SMPWrapperBase(nn.Module):
    """
    Base wrapper class for SMP models
    Handles 2.5D input (3 channels) and binary output (1 channel)
    + Optional attention injection
    """
    
    def __init__(self, config, architecture):
        super().__init__()
        
        if not SMP_AVAILABLE:
            raise ImportError(
                "segmentation_models_pytorch is required but not installed.\n"
                "Install with: pip install segmentation-models-pytorch"
            )
        
        self.config = config
        self.architecture = architecture
        
        # Get configuration
        encoder_name = getattr(config, 'ENCODER_NAME', 'resnet34')
        encoder_weights = getattr(config, 'ENCODER_WEIGHTS', 'imagenet')
        in_channels = config.IN_CHANNELS
        out_channels = config.OUT_CHANNELS
        
        # Create model using SMP
        self.model = self._create_model(
            architecture=architecture,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels
        )
        
        # Inject attention if enabled
        use_se = getattr(config, 'USE_SE_ATTENTION', False)
        use_cbam = getattr(config, 'USE_CBAM_ATTENTION', False)
        use_eca = getattr(config, 'USE_ECA_ATTENTION', False)
        
        if any([use_se, use_cbam, use_eca]) and ATTENTION_AVAILABLE:
            self.model = AttentionInjector(self.model, config)
    
    def _create_model(self, architecture, encoder_name, encoder_weights, in_channels, classes):
        """Override in subclasses"""
        raise NotImplementedError
    
    def forward(self, x):
        """Forward pass through SMP model"""
        logits = self.model(x)
        # Apply sigmoid for binary segmentation
        return torch.sigmoid(logits)


class UNetPlusPlusWrapper(SMPWrapperBase):
    """
    U-Net++ (Nested U-Net) wrapper
    
    Features:
    - Dense skip connections between encoder and decoder
    - Multiple depths for multi-scale feature fusion
    - Better gradient flow than standard U-Net
    
    Paper: "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
    """
    
    def __init__(self, config):
        super().__init__(config, 'unet++')
    
    def _create_model(self, architecture, encoder_name, encoder_weights, in_channels, classes):
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,  # We apply sigmoid separately
        )


class FPNWrapper(SMPWrapperBase):
    """
    Feature Pyramid Network (FPN) wrapper
    
    Features:
    - Multi-scale feature pyramid
    - Top-down pathway with lateral connections
    - Excellent for objects at different scales
    
    Paper: "Feature Pyramid Networks for Object Detection"
    """
    
    def __init__(self, config):
        super().__init__(config, 'fpn')
    
    def _create_model(self, architecture, encoder_name, encoder_weights, in_channels, classes):
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )


class DeepLabV3PlusWrapper(SMPWrapperBase):
    """
    DeepLabV3+ wrapper
    
    Features:
    - Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context
    - Encoder-decoder structure with atrous convolutions
    - Excellent boundary detection
    
    Paper: "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
    """
    
    def __init__(self, config):
        super().__init__(config, 'deeplabv3+')
    
    def _create_model(self, architecture, encoder_name, encoder_weights, in_channels, classes):
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )


class MANetWrapper(SMPWrapperBase):
    """
    Multi-Attention Network (MANet) wrapper
    
    Features:
    - Position attention and channel attention modules
    - Captures both spatial and channel-wise relationships
    - Strong performance on medical images
    
    Paper: "Multi-Attention Network for Semantic Segmentation"
    """
    
    def __init__(self, config):
        super().__init__(config, 'manet')
    
    def _create_model(self, architecture, encoder_name, encoder_weights, in_channels, classes):
        return smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )


class PSPNetWrapper(SMPWrapperBase):
    """
    Pyramid Scene Parsing Network (PSPNet) wrapper
    
    Features:
    - Pyramid pooling module for multi-scale context aggregation
    - Captures global context at multiple scales
    - Strong for scene understanding
    
    Paper: "Pyramid Scene Parsing Network"
    """
    
    def __init__(self, config):
        super().__init__(config, 'pspnet')
    
    def _create_model(self, architecture, encoder_name, encoder_weights, in_channels, classes):
        return smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )


# Additional architectures can be added easily:

class LinkNetWrapper(SMPWrapperBase):
    """
    LinkNet wrapper
    
    Features:
    - Efficient encoder-decoder architecture
    - Fast inference time
    - Good for real-time applications
    """
    
    def __init__(self, config):
        super().__init__(config, 'linknet')
    
    def _create_model(self, architecture, encoder_name, encoder_weights, in_channels, classes):
        return smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )


class PanWrapper(SMPWrapperBase):
    """
    Path Aggregation Network (PAN) wrapper
    
    Features:
    - Enhanced FPN with bottom-up path
    - Better information flow
    """
    
    def __init__(self, config):
        super().__init__(config, 'pan')
    
    def _create_model(self, architecture, encoder_name, encoder_weights, in_channels, classes):
        return smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )


# Utility functions

def list_available_encoders():
    """List all available encoders from SMP"""
    if not SMP_AVAILABLE:
        print("segmentation_models_pytorch not installed")
        return []
    
    encoders = smp.encoders.get_encoder_names()
    return sorted(encoders)


def get_encoder_info(encoder_name):
    """Get information about a specific encoder"""
    if not SMP_AVAILABLE:
        return None
    
    try:
        from segmentation_models_pytorch.encoders import get_encoder
        encoder = get_encoder(encoder_name, in_channels=3, depth=5, weights=None)
        
        params = sum(p.numel() for p in encoder.parameters())
        
        return {
            'name': encoder_name,
            'parameters': params,
            'parameters_M': params / 1e6,
        }
    except Exception as e:
        return {'error': str(e)}


if __name__ == "__main__":
    # Test SMP availability
    print("="*70)
    print("Testing SMP Wrappers")
    print("="*70)
    
    if SMP_AVAILABLE:
        print("✅ segmentation_models_pytorch is installed")
        
        # List available encoders
        print("\n📋 Available Encoders:")
        encoders = list_available_encoders()
        
        # Show recommended encoders
        recommended = ['resnet34', 'resnet50', 'efficientnet-b0', 'efficientnet-b3']
        print("\nRecommended encoders for medical imaging:")
        for enc in recommended:
            if enc in encoders:
                info = get_encoder_info(enc)
                print(f"  - {enc}: ~{info['parameters_M']:.1f}M params")
        
    else:
        print("❌ segmentation_models_pytorch is NOT installed")
        print("\nInstall with:")
        print("  pip install segmentation-models-pytorch")
