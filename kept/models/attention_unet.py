"""
Attention U-Net Model for DWI Ischemic Stroke Segmentation
Implements 2.5D Attention U-Net with skip connections and attention gates
+ Optional additional attention mechanisms (SE, CBAM, Dual, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import additional attention modules
try:
    from .attention_modules import SEBlock, CBAM, ECABlock, DualAttention, MultiScaleAttention
except ImportError:
    # Fallback if attention_modules not available
    SEBlock = CBAM = ECABlock = DualAttention = MultiScaleAttention = None


class ConvBlock(nn.Module):
    """
    Convolutional Block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    + Optional attention modules (SE, CBAM, ECA)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dropout: Dropout probability (default: 0.0)
        use_se: Use SE attention block
        use_cbam: Use CBAM attention block
        use_eca: Use ECA attention block
    """
    
    def __init__(self, in_channels, out_channels, dropout=0.0, 
                 use_se=False, use_cbam=False, use_eca=False):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
        # Optional attention modules
        self.use_se = use_se and SEBlock is not None
        self.use_cbam = use_cbam and CBAM is not None
        self.use_eca = use_eca and ECABlock is not None
        
        if self.use_se:
            self.se = SEBlock(out_channels)
        if self.use_cbam:
            self.cbam = CBAM(out_channels)
        if self.use_eca:
            self.eca = ECABlock(out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        if self.dropout:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Apply attention modules if enabled
        if self.use_se:
            x = self.se(x)
        if self.use_cbam:
            x = self.cbam(x)
        if self.use_eca:
            x = self.eca(x)
        
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
    
    def __init__(self, in_channels, out_channels, dropout=0.0, 
                 use_se=False, use_cbam=False, use_eca=False):
        super(EncoderBlock, self).__init__()
        
        self.conv = ConvBlock(in_channels, out_channels, dropout, use_se, use_cbam, use_eca)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        """Returns: x_conv (for skip), x_pool (for next encoder)"""
        x_conv = self.conv(x)
        x_pool = self.pool(x_conv)
        return x_conv, x_pool


class DecoderBlock(nn.Module):
    """Decoder block: Upsampling + Attention Gate + Concatenation + ConvBlock"""
    
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True, dropout=0.0,
                 use_se=False, use_cbam=False, use_eca=False):
        super(DecoderBlock, self).__init__()
        
        self.use_attention = use_attention
        
        # Upsampling
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Attention gate
        if use_attention:
            self.attention = AttentionGate(skip_channels, in_channels // 2)
        
        # Convolutional block
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels, dropout,
                            use_se, use_cbam, use_eca)
    
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
        
        # Get additional attention configs
        use_se = getattr(config, 'USE_SE_ATTENTION', False)
        use_cbam = getattr(config, 'USE_CBAM_ATTENTION', False)
        use_eca = getattr(config, 'USE_ECA_ATTENTION', False)
        use_dual = getattr(config, 'USE_DUAL_ATTENTION', False)
        use_multiscale = getattr(config, 'USE_MULTISCALE_ATTENTION', False)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        
        # ==================== Encoder ====================
        self.encoders = nn.ModuleList()
        
        prev_channels = in_channels
        for channels in encoder_channels:
            self.encoders.append(EncoderBlock(prev_channels, channels, dropout,
                                            use_se, use_cbam, use_eca))
            prev_channels = channels
        
        # ==================== Bottleneck ====================
        self.bottleneck = ConvBlock(encoder_channels[-1], bottleneck_channels, dropout,
                                   use_se, use_cbam, use_eca)
        
        # Add Dual Attention or Multi-Scale Attention to bottleneck if enabled
        self.bottleneck_attention = None
        if use_dual and DualAttention is not None:
            self.bottleneck_attention = DualAttention(bottleneck_channels)
            print(f"   🔥 Added Dual Attention to bottleneck ({bottleneck_channels} channels)")
        elif use_multiscale and MultiScaleAttention is not None:
            self.bottleneck_attention = MultiScaleAttention(bottleneck_channels)
            print(f"   🔥 Added Multi-Scale Attention to bottleneck ({bottleneck_channels} channels)")
        
        # ==================== Decoder ====================
        self.decoders = nn.ModuleList()
        
        decoder_channels = encoder_channels[::-1]  # Reverse order
        
        prev_channels = bottleneck_channels
        for i, channels in enumerate(decoder_channels):
            skip_channels = channels
            out_ch = channels if i < len(decoder_channels) - 1 else decoder_channels[-1]
            
            self.decoders.append(
                DecoderBlock(prev_channels, skip_channels, out_ch, use_attention, dropout,
                           use_se, use_cbam, use_eca)
            )
            prev_channels = out_ch
        
        # ==================== Output ====================
        self.output = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Print attention summary
        self._print_attention_summary(use_se, use_cbam, use_eca, use_dual, use_multiscale)
        # Print attention summary
        self._print_attention_summary(use_se, use_cbam, use_eca, use_dual, use_multiscale)
    
    def _print_attention_summary(self, use_se, use_cbam, use_eca, use_dual, use_multiscale):
        """Print summary of enabled attention mechanisms"""
        attention_enabled = []
        if self.use_attention:
            attention_enabled.append("Spatial Attention Gates")
        if use_se:
            attention_enabled.append("SE (Squeeze-Excitation)")
        if use_cbam:
            attention_enabled.append("CBAM (Channel+Spatial)")
        if use_eca:
            attention_enabled.append("ECA (Efficient Channel)")
        if use_dual:
            attention_enabled.append("Dual Attention (Position+Channel)")
        if use_multiscale:
            attention_enabled.append("Multi-Scale Attention")
        
        if attention_enabled:
            print(f"\n   🎯 Active Attention Mechanisms:")
            for att in attention_enabled:
                print(f"      ✅ {att}")
    
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
        
        # Apply bottleneck attention if enabled
        if self.bottleneck_attention is not None:
            x = self.bottleneck_attention(x)
        
        # Decoder Path
        skip_connections = skip_connections[::-1]
        
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
        
        # Output
        x = self.output(x)
        
        return x


# ==================== Attention U-Net with Deep Supervision ====================

class AttentionUNetDeepSupervision(nn.Module):
    """
    Attention U-Net with Deep Supervision for Medical Image Segmentation
    
    Deep Supervision adds auxiliary outputs at intermediate decoder layers to:
    - Improve gradient flow to earlier layers
    - Provide multi-scale supervision signals
    - Speed up convergence
    - Improve final segmentation quality
    
    Architecture:
        - Same as AttentionUNet but with additional output heads at each decoder level
        - Returns: [main_output, aux_output1, aux_output2, aux_output3, ...]
        - During training: Combined loss from all outputs (weighted)
        - During inference: Use only main_output
    
    Args:
        Same as AttentionUNet, plus:
        num_supervision_levels: Number of auxiliary outputs (default: 3)
                                Typically matches number of decoder blocks
    """
    
    def __init__(self, in_channels=3, out_channels=1,
                 encoder_channels=[64, 128, 256, 512],
                 bottleneck_channels=1024,
                 use_attention=True,
                 dropout=0.0,
                 use_se=False,
                 use_cbam=False,
                 use_eca=False,
                 use_dual=False,
                 use_multiscale=False,
                 num_supervision_levels=3):
        super(AttentionUNetDeepSupervision, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channels = encoder_channels
        self.bottleneck_channels = bottleneck_channels
        self.use_attention = use_attention
        self.num_supervision_levels = min(num_supervision_levels, len(encoder_channels))
        
        print(f"\n🏗️  Building Attention U-Net with Deep Supervision...")
        print(f"   Input channels: {in_channels}")
        print(f"   Output channels: {out_channels}")
        print(f"   Encoder channels: {encoder_channels}")
        print(f"   Bottleneck channels: {bottleneck_channels}")
        print(f"   🔥 Deep Supervision Levels: {self.num_supervision_levels}")
        print(f"   Attention Gates: {'✅' if use_attention else '❌'}")
        print(f"   Dropout: {dropout}")
        
        # ==================== Encoder ====================
        self.encoders = nn.ModuleList()
        
        prev_channels = in_channels
        for channels in encoder_channels:
            self.encoders.append(
                EncoderBlock(prev_channels, channels, dropout,
                           use_se, use_cbam, use_eca)
            )
            prev_channels = channels
        
        # ==================== Bottleneck ====================
        self.bottleneck = ConvBlock(encoder_channels[-1], bottleneck_channels, dropout,
                                   use_se, use_cbam, use_eca)
        
        # Add Dual Attention or Multi-Scale Attention to bottleneck if enabled
        self.bottleneck_attention = None
        if use_dual and DualAttention is not None:
            self.bottleneck_attention = DualAttention(bottleneck_channels)
            print(f"   🔥 Added Dual Attention to bottleneck ({bottleneck_channels} channels)")
        elif use_multiscale and MultiScaleAttention is not None:
            self.bottleneck_attention = MultiScaleAttention(bottleneck_channels)
            print(f"   🔥 Added Multi-Scale Attention to bottleneck ({bottleneck_channels} channels)")
        
        # ==================== Decoder ====================
        self.decoders = nn.ModuleList()
        
        decoder_channels = encoder_channels[::-1]  # Reverse order
        
        prev_channels = bottleneck_channels
        for i, channels in enumerate(decoder_channels):
            skip_channels = channels
            out_ch = channels if i < len(decoder_channels) - 1 else decoder_channels[-1]
            
            self.decoders.append(
                DecoderBlock(prev_channels, skip_channels, out_ch, use_attention, dropout,
                           use_se, use_cbam, use_eca)
            )
            prev_channels = out_ch
        
        # ==================== Main Output ====================
        self.main_output = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # ==================== Deep Supervision Outputs ====================
        # Create auxiliary output heads for intermediate decoder levels
        self.aux_outputs = nn.ModuleList()
        
        for i in range(self.num_supervision_levels):
            # Get channels from decoder at this level
            if i < len(decoder_channels):
                aux_in_channels = decoder_channels[i]
            else:
                aux_in_channels = decoder_channels[-1]
            
            # Create auxiliary output head
            aux_head = nn.Sequential(
                nn.Conv2d(aux_in_channels, out_channels, kernel_size=1),
                nn.Sigmoid()
            )
            self.aux_outputs.append(aux_head)
            print(f"   📊 Auxiliary Output {i+1}: {aux_in_channels} → {out_channels} channels")
        
        # Print attention summary
        self._print_attention_summary(use_se, use_cbam, use_eca, use_dual, use_multiscale)
    
    def _print_attention_summary(self, use_se, use_cbam, use_eca, use_dual, use_multiscale):
        """Print summary of enabled attention mechanisms"""
        attention_enabled = []
        if self.use_attention:
            attention_enabled.append("Spatial Attention Gates")
        if use_se:
            attention_enabled.append("SE (Squeeze-Excitation)")
        if use_cbam:
            attention_enabled.append("CBAM (Channel+Spatial)")
        if use_eca:
            attention_enabled.append("ECA (Efficient Channel)")
        if use_dual:
            attention_enabled.append("Dual Attention (Position+Channel)")
        if use_multiscale:
            attention_enabled.append("Multi-Scale Attention")
        
        if attention_enabled:
            print(f"\n   🎯 Active Attention Mechanisms:")
            for att in attention_enabled:
                print(f"      ✅ {att}")
    
    def forward(self, x, return_aux=True):
        """
        Args:
            x: Input tensor (B, 3, H, W) for 2.5D input
            return_aux: If True, return all outputs for deep supervision
                       If False, return only main output (for inference)
        
        Returns:
            If return_aux=True:
                List of [main_output, aux_output_1, aux_output_2, ...]
            If return_aux=False:
                main_output only
        """
        input_size = x.shape[2:]  # (H, W)
        
        # Encoder Path
        skip_connections = []
        
        for encoder in self.encoders:
            x_skip, x = encoder(x)
            skip_connections.append(x_skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Apply bottleneck attention if enabled
        if self.bottleneck_attention is not None:
            x = self.bottleneck_attention(x)
        
        # Decoder Path with Deep Supervision
        skip_connections = skip_connections[::-1]
        aux_outputs = []
        
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
            
            # Generate auxiliary output at this level (if needed)
            if return_aux and i < self.num_supervision_levels:
                aux_out = self.aux_outputs[i](x)
                # Upsample to match input size
                if aux_out.shape[2:] != input_size:
                    aux_out = F.interpolate(aux_out, size=input_size, 
                                          mode='bilinear', align_corners=True)
                aux_outputs.append(aux_out)
        
        # Main Output
        main_out = self.main_output(x)
        
        # Return based on mode
        if return_aux:
            # Return [main_output, aux1, aux2, ...] for training
            return [main_out] + aux_outputs
        else:
            # Return only main output for inference
            return main_out

