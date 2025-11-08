"""
Architecture Comparison Tool
Compare different segmentation architectures on the DWI dataset
"""

import torch
import config
from models import get_model
from pathlib import Path


def print_architecture_info():
    """Print information about all available architectures"""
    
    print("="*80)
    print("🏗️  DWI SEGMENTATION - AVAILABLE ARCHITECTURES")
    print("="*80)
    
    architectures = {
        'attention_unet': {
            'name': 'Attention U-Net',
            'description': 'Custom U-Net with attention gates (current baseline)',
            'params_approx': '17.5M (Medium) / 31M (Large)',
            'speed': 'Fast (~7s/epoch)',
            'memory': 'Medium (~3.5GB)',
            'best_for': 'Baseline, proven performance',
            'paper': 'Oktay et al., 2018'
        },
        'unet++': {
            'name': 'U-Net++',
            'description': 'Nested U-Net with dense skip connections',
            'params_approx': '~20M (ResNet34)',
            'speed': 'Medium (~8.5s/epoch)',
            'memory': 'Medium-High (~4.2GB)',
            'best_for': 'Better gradient flow, multi-scale features',
            'paper': 'Zhou et al., 2018'
        },
        'fpn': {
            'name': 'Feature Pyramid Network',
            'description': 'Multi-scale feature pyramid with lateral connections',
            'params_approx': '~25M (ResNet34)',
            'speed': 'Medium (~7.8s/epoch)',
            'memory': 'Medium (~3.8GB)',
            'best_for': 'Objects at different scales',
            'paper': 'Lin et al., 2017'
        },
        'deeplabv3+': {
            'name': 'DeepLabV3+',
            'description': 'ASPP module for multi-scale context',
            'params_approx': '~40M (ResNet34)',
            'speed': 'Slow (~10s/epoch)',
            'memory': 'High (~5.5GB)',
            'best_for': 'Boundary detection, dense predictions',
            'paper': 'Chen et al., 2018'
        },
        'manet': {
            'name': 'MANet',
            'description': 'Multi-attention network (position + channel)',
            'params_approx': '~22M (ResNet34)',
            'speed': 'Medium (~9s/epoch)',
            'memory': 'Medium-High (~4.5GB)',
            'best_for': 'Medical images, attention mechanisms',
            'paper': 'Fan et al., 2020'
        },
        'pspnet': {
            'name': 'PSPNet',
            'description': 'Pyramid scene parsing with multi-scale pooling',
            'params_approx': '~45M (ResNet34)',
            'speed': 'Slow (~11s/epoch)',
            'memory': 'High (~6.0GB)',
            'best_for': 'Global context, scene understanding',
            'paper': 'Zhao et al., 2017'
        }
    }
    
    # Print detailed information
    for arch_key, arch_info in architectures.items():
        print(f"\n{'─'*80}")
        print(f"📦 {arch_info['name']} ({arch_key})")
        print(f"{'─'*80}")
        print(f"Description: {arch_info['description']}")
        print(f"Parameters:  {arch_info['params_approx']}")
        print(f"Speed:       {arch_info['speed']}")
        print(f"Memory:      {arch_info['memory']}")
        print(f"Best for:    {arch_info['best_for']}")
        print(f"Paper:       {arch_info['paper']}")
    
    print(f"\n{'='*80}")
    print("\n📝 USAGE:")
    print("   Edit config.py and set:")
    print("   MODEL_ARCHITECTURE = 'attention_unet'  # or any key above")
    print("   ENCODER_NAME = 'resnet34'  # for SMP models")
    print("   ENCODER_WEIGHTS = 'imagenet'  # or None")
    print(f"\n{'='*80}\n")


def test_architecture(arch_name, encoder_name='resnet34', encoder_weights='imagenet'):
    """Test if an architecture can be loaded and run forward pass"""
    
    print(f"\n{'='*80}")
    print(f"🧪 Testing Architecture: {arch_name}")
    print(f"{'='*80}")
    
    # Save original config
    original_arch = config.MODEL_ARCHITECTURE
    original_encoder = getattr(config, 'ENCODER_NAME', 'resnet34')
    original_weights = getattr(config, 'ENCODER_WEIGHTS', 'imagenet')
    
    try:
        # Set config
        config.MODEL_ARCHITECTURE = arch_name
        config.ENCODER_NAME = encoder_name
        config.ENCODER_WEIGHTS = encoder_weights
        
        # Create model
        print(f"\n1️⃣  Creating model...")
        model = get_model(config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n📊 Model Statistics:")
        print(f"   Total params: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"   Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"   Model size: ~{total_params * 4 / (1024**2):.2f} MB (FP32)")
        
        # Test forward pass
        print(f"\n2️⃣  Testing forward pass...")
        model.eval()
        
        with torch.no_grad():
            # Test different input sizes
            test_inputs = [
                (1, 3, 256, 256),  # Single sample, standard size
                (2, 3, 256, 256),  # Small batch
                (4, 3, 128, 128),  # Smaller image
            ]
            
            for batch_size, channels, h, w in test_inputs:
                x = torch.randn(batch_size, channels, h, w)
                y = model(x)
                
                # Check output shape
                expected_shape = (batch_size, config.OUT_CHANNELS, h, w)
                assert y.shape == expected_shape, f"Output shape mismatch! Expected {expected_shape}, got {y.shape}"
                
                # Check output range
                assert y.min() >= 0 and y.max() <= 1, f"Output not in [0,1]! Range: [{y.min():.3f}, {y.max():.3f}]"
                
                print(f"   ✅ Input {x.shape} → Output {y.shape} [range: {y.min():.3f}, {y.max():.3f}]")
        
        # Test gradient flow
        print(f"\n3️⃣  Testing gradient flow...")
        model.train()
        
        x = torch.randn(2, 3, 128, 128, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 128, 128)).float()
        
        y = model(x)
        loss = torch.nn.functional.binary_cross_entropy(y, target)
        loss.backward()
        
        has_grad = x.grad is not None
        grad_mean = x.grad.mean().item() if has_grad else 0.0
        
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Gradient flow: {'✅ OK' if has_grad else '❌ FAILED'}")
        print(f"   Gradient mean: {grad_mean:.6f}")
        
        # Memory estimation
        print(f"\n4️⃣  Memory estimation (batch_size=16, 256×256):")
        batch_size = 16
        input_size = (256, 256)
        
        input_mem = batch_size * 3 * input_size[0] * input_size[1] * 4 / (1024**2)
        model_mem = total_params * 4 / (1024**2)
        activation_mem = input_mem * 10  # Rough estimate
        grad_mem = model_mem
        optimizer_mem = model_mem * 2
        
        total_mem = input_mem + model_mem + activation_mem + grad_mem + optimizer_mem
        
        print(f"   Input:       {input_mem:.2f} MB")
        print(f"   Model:       {model_mem:.2f} MB")
        print(f"   Activations: {activation_mem:.2f} MB")
        print(f"   Gradients:   {grad_mem:.2f} MB")
        print(f"   Optimizer:   {optimizer_mem:.2f} MB")
        print(f"   Total:       {total_mem:.2f} MB (~{total_mem/1024:.2f} GB)")
        print(f"   Recommended GPU: >= {total_mem * 1.5 / 1024:.1f} GB VRAM")
        
        print(f"\n{'='*80}")
        print(f"✅ {arch_name.upper()} - ALL TESTS PASSED!")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing {arch_name}:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original config
        config.MODEL_ARCHITECTURE = original_arch
        config.ENCODER_NAME = original_encoder
        config.ENCODER_WEIGHTS = original_weights


def test_all_architectures():
    """Test all available architectures"""
    
    print("\n" + "="*80)
    print("🧪 TESTING ALL ARCHITECTURES")
    print("="*80 + "\n")
    
    architectures = [
        'attention_unet',
        'unet++',
        'fpn',
        'deeplabv3+',
        'manet',
        'pspnet'
    ]
    
    results = {}
    
    for arch in architectures:
        if arch == 'attention_unet':
            # Test with original config
            results[arch] = test_architecture(arch)
        else:
            # Test with default encoder
            results[arch] = test_architecture(arch, encoder_name='resnet34', encoder_weights=None)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for arch, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {arch:20s} {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n   Total: {passed}/{total} architectures passed")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'list' or command == 'info':
            print_architecture_info()
            
        elif command == 'test':
            if len(sys.argv) > 2:
                arch = sys.argv[2]
                test_architecture(arch)
            else:
                test_all_architectures()
        
        else:
            print("Unknown command. Use: python compare_architectures.py [list|test|test <arch>]")
    
    else:
        # Default: show info
        print_architecture_info()
        
        # Ask if user wants to test
        try:
            response = input("\n🧪 Do you want to test all architectures? (y/n): ").strip().lower()
            if response == 'y':
                test_all_architectures()
        except KeyboardInterrupt:
            print("\n\nAborted by user.")
