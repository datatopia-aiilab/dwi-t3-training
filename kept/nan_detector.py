"""
NaN Detection and Recovery Script
Helps detect and prevent NaN loss during training
"""

import torch
import torch.nn as nn


class NaNDetector:
    """Detect NaN in model outputs and gradients"""
    
    def __init__(self, model):
        self.model = model
        self.nan_detected = False
        
    def check_output(self, output, name="output"):
        """Check if output contains NaN or Inf"""
        if torch.isnan(output).any():
            print(f"❌ NaN detected in {name}")
            self.nan_detected = True
            return True
        if torch.isinf(output).any():
            print(f"⚠️  Inf detected in {name}")
            return True
        return False
    
    def check_gradients(self):
        """Check all model gradients for NaN"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"❌ NaN gradient in: {name}")
                    self.nan_detected = True
                    return True
                if torch.isinf(param.grad).any():
                    print(f"⚠️  Inf gradient in: {name}")
                    return True
        return False
    
    def check_parameters(self):
        """Check all model parameters for NaN"""
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"❌ NaN parameter in: {name}")
                self.nan_detected = True
                return True
            if torch.isinf(param).any():
                print(f"⚠️  Inf parameter in: {name}")
                return True
        return False


def add_nan_hooks(model, verbose=False):
    """
    Add hooks to detect NaN during forward/backward pass
    
    Usage:
        hooks = add_nan_hooks(model, verbose=True)
        # ... training ...
        # Remove hooks when done
        for hook in hooks:
            hook.remove()
    """
    
    hooks = []
    
    def forward_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                print(f"❌ NaN in forward pass: {module.__class__.__name__}")
            elif verbose:
                print(f"✅ {module.__class__.__name__}: OK")
    
    def backward_hook(module, grad_input, grad_output):
        if isinstance(grad_output[0], torch.Tensor):
            if torch.isnan(grad_output[0]).any():
                print(f"❌ NaN in backward pass: {module.__class__.__name__}")
    
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(forward_hook))
        hooks.append(module.register_backward_hook(backward_hook))
    
    return hooks


def clip_gradients_safe(model, max_norm=0.5):
    """
    Safely clip gradients with NaN detection
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
    
    Returns:
        total_norm: Total gradient norm before clipping
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    
    if len(parameters) == 0:
        return 0.0
    
    # Check for NaN gradients first
    has_nan = False
    for p in parameters:
        if torch.isnan(p.grad).any():
            has_nan = True
            print(f"❌ NaN gradient detected! Setting to zero.")
            p.grad.zero_()
    
    if has_nan:
        return 0.0
    
    # Compute total norm
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2
    )
    
    # Clip if needed
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
    
    return total_norm.item()


def check_attention_weights(model, threshold=10.0):
    """
    Check if attention weights are becoming too large
    
    Args:
        model: Model with attention modules
        threshold: Warning threshold for large weights
    
    Returns:
        warnings: List of warnings
    """
    warnings = []
    
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'cbam' in name.lower() or 'se' in name.lower():
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    max_val = param.abs().max().item()
                    if max_val > threshold:
                        warnings.append(f"{name}.{param_name}: max={max_val:.2f}")
    
    return warnings


# Recommendation for training with attention
TRAINING_TIPS = """
🎯 TIPS TO PREVENT NaN WITH ATTENTION MECHANISMS:

1. **Lower Learning Rate**
   - Start with LR = 1e-5 or lower
   - Gradually increase if stable

2. **Gradient Clipping**
   - Use GRADIENT_CLIP_VALUE = 0.5 or lower
   - Monitor gradient norms

3. **Batch Normalization**
   - Already added to attention modules
   - Helps stabilize training

4. **Residual Scaling**
   - CBAM and SE now use scale=0.1
   - Prevents large attention effects

5. **Mixed Precision Training**
   - Use with caution with attention
   - May need to disable if unstable

6. **Warmup Strategy**
   - Start without attention
   - Add attention after few epochs
   - Or use very low learning rate initially

7. **Monitor**
   - Check loss every epoch
   - If NaN appears, reduce LR immediately
   - Consider restarting from last checkpoint

8. **Alternative: Use SE instead of CBAM**
   - SE is lighter and more stable
   - USE_SE_ATTENTION = True
   - USE_CBAM_ATTENTION = False
"""


if __name__ == "__main__":
    print(TRAINING_TIPS)
