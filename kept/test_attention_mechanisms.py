"""
Test Attention Mechanisms
Verify that all attention modules work correctly with all architectures
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from models import get_model


def test_attention_combinations():
    """Test all attention mechanism combinations"""
    
    print("\n" + "="*70)
    print("🧪 TESTING ATTENTION MECHANISMS")
    print("="*70)
    
    # Architectures to test
    architectures = [
        'attention_unet',
        'unet++',
        'fpn',
        'deeplabv3+',
        'manet',
        'pspnet'
    ]
    
    # Attention combinations to test
    attention_configs = [
        {'name': 'None', 'se': False, 'cbam': False, 'eca': False, 'dual': False, 'multiscale': False},
        {'name': 'SE Only', 'se': True, 'cbam': False, 'eca': False, 'dual': False, 'multiscale': False},
        {'name': 'CBAM Only', 'se': False, 'cbam': True, 'eca': False, 'dual': False, 'multiscale': False},
        {'name': 'ECA Only', 'se': False, 'cbam': False, 'eca': True, 'dual': False, 'multiscale': False},
        {'name': 'SE + CBAM', 'se': True, 'cbam': True, 'eca': False, 'dual': False, 'multiscale': False},
        {'name': 'SE + Dual', 'se': True, 'cbam': False, 'eca': False, 'dual': True, 'multiscale': False},
        {'name': 'CBAM + Multi-Scale', 'se': False, 'cbam': True, 'eca': False, 'dual': False, 'multiscale': True},
        {'name': 'All (SE+CBAM+Dual+MS)', 'se': True, 'cbam': True, 'eca': False, 'dual': True, 'multiscale': True},
    ]
    
    # Test input
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 384, 384)
    
    results = []
    
    for arch in architectures:
        print(f"\n{'='*70}")
        print(f"🏗️  Testing: {arch.upper()}")
        print(f"{'='*70}")
        
        for att_config in attention_configs:
            try:
                # Set architecture
                config.MODEL_ARCHITECTURE = arch
                
                # Set attention configs
                config.USE_SE_ATTENTION = att_config['se']
                config.USE_CBAM_ATTENTION = att_config['cbam']
                config.USE_ECA_ATTENTION = att_config['eca']
                config.USE_DUAL_ATTENTION = att_config['dual']
                config.USE_MULTISCALE_ATTENTION = att_config['multiscale']
                
                print(f"\n   📋 Config: {att_config['name']}")
                
                # Create model
                model = get_model(config)
                model.eval()
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print(f"      Total params: {total_params:,}")
                print(f"      Trainable params: {trainable_params:,}")
                
                # Forward pass
                with torch.no_grad():
                    output = model(test_input)
                
                # Verify output shape
                expected_shape = (batch_size, 1, 384, 384)
                if output.shape == expected_shape:
                    print(f"      ✅ Output shape correct: {output.shape}")
                else:
                    print(f"      ❌ Output shape incorrect: {output.shape} (expected {expected_shape})")
                
                # Verify output range [0, 1]
                out_min, out_max = output.min().item(), output.max().item()
                if 0 <= out_min and out_max <= 1:
                    print(f"      ✅ Output range valid: [{out_min:.4f}, {out_max:.4f}]")
                else:
                    print(f"      ⚠️  Output range: [{out_min:.4f}, {out_max:.4f}]")
                
                # Memory usage
                memory_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
                
                results.append({
                    'architecture': arch,
                    'attention': att_config['name'],
                    'params': total_params,
                    'success': True,
                    'memory_mb': memory_mb
                })
                
                print(f"      ✅ Test PASSED")
                
            except Exception as e:
                print(f"      ❌ Test FAILED: {e}")
                results.append({
                    'architecture': arch,
                    'attention': att_config['name'],
                    'params': 0,
                    'success': False,
                    'memory_mb': 0,
                    'error': str(e)
                })
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for arch in architectures:
        arch_results = [r for r in results if r['architecture'] == arch]
        passed = sum(1 for r in arch_results if r['success'])
        total = len(arch_results)
        
        print(f"\n{arch.upper()}:")
        print(f"   Passed: {passed}/{total}")
        
        if passed < total:
            failed = [r for r in arch_results if not r['success']]
            for f in failed:
                print(f"      ❌ {f['attention']}: {f.get('error', 'Unknown error')}")
    
    # Overall statistics
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    
    print(f"\n{'='*70}")
    print(f"OVERALL: {passed_tests}/{total_tests} tests passed ({100*passed_tests/total_tests:.1f}%)")
    print(f"{'='*70}")
    
    return results


def test_parameter_counts():
    """Compare parameter counts with different attention mechanisms"""
    
    print("\n" + "="*70)
    print("📊 PARAMETER COUNT ANALYSIS")
    print("="*70)
    
    arch = 'attention_unet'
    config.MODEL_ARCHITECTURE = arch
    
    configs = [
        ('Baseline', False, False, False, False, False),
        ('+ SE', True, False, False, False, False),
        ('+ CBAM', False, True, False, False, False),
        ('+ ECA', False, False, True, False, False),
        ('+ SE + CBAM', True, True, False, False, False),
        ('+ SE + Dual', True, False, False, True, False),
        ('+ All', True, True, False, True, True),
    ]
    
    baseline_params = None
    
    for name, se, cbam, eca, dual, ms in configs:
        config.USE_SE_ATTENTION = se
        config.USE_CBAM_ATTENTION = cbam
        config.USE_ECA_ATTENTION = eca
        config.USE_DUAL_ATTENTION = dual
        config.USE_MULTISCALE_ATTENTION = ms
        
        model = get_model(config)
        params = sum(p.numel() for p in model.parameters())
        
        if baseline_params is None:
            baseline_params = params
            overhead = 0
        else:
            overhead = params - baseline_params
        
        overhead_pct = 100 * overhead / baseline_params if baseline_params > 0 else 0
        
        print(f"\n{name:20s}: {params:12,} params  (+{overhead:8,}  +{overhead_pct:5.2f}%)")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    # Test all combinations
    results = test_attention_combinations()
    
    # Parameter analysis
    test_parameter_counts()
    
    print("\n✅ All tests completed!\n")
