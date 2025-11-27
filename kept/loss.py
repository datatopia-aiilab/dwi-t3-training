"""
Loss Functions for DWI Ischemic Stroke Segmentation
Implements Focal Loss, Dice Loss, and Combo Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and focusing on hard examples
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weight for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
               Higher gamma = more focus on hard examples
        smooth: Smoothing factor to avoid log(0)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, 1, H, W) with values in [0, 1] (after sigmoid)
            target: Ground truth (B, 1, H, W) with values in {0, 1}
        
        Returns:
            loss: Scalar focal loss value
        """
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, self.smooth, 1.0 - self.smooth)
        
        # Calculate BCE
        bce = - (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        
        # Calculate focal weight
        # p_t = p if y=1, else 1-p
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Calculate alpha weight
        # alpha_t = alpha if y=1, else 1-alpha
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Focal loss
        focal_loss = alpha_t * focal_weight * bce
        
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    
    Measures the overlap between prediction and ground truth
    
    Formula: DiceLoss = 1 - DiceScore
             DiceScore = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    
    Args:
        smooth: Smoothing factor to avoid division by zero
    """
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, 1, H, W) with values in [0, 1]
            target: Ground truth (B, 1, H, W) with values in {0, 1}
        
        Returns:
            loss: Scalar dice loss value (1 - dice_score)
        """
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        # Calculate Dice score
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss = 1 - Dice score
        dice_loss = 1.0 - dice_score
        
        return dice_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice Loss
    
    Allows weighting of false positives and false negatives differently
    Useful when you want to prioritize recall over precision or vice versa
    
    Args:
        alpha: Weight for false positives
        beta: Weight for false negatives
        smooth: Smoothing factor
        
    Note: alpha=beta=0.5 reduces to Dice Loss
    """
    
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, 1, H, W)
            target: Ground truth (B, 1, H, W)
        
        Returns:
            loss: Scalar Tversky loss
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()
        
        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1.0 - tversky


class ComboLoss(nn.Module):
    """
    Combination of Focal Loss and Dice Loss
    
    This is the CORE loss function for our project!
    
    Strategy:
        - Focal Loss: Handles class imbalance and focuses on hard examples (faint lesions)
        - Dice Loss: Directly optimizes overlap/segmentation quality
        - Combo: Gets the best of both worlds
    
    Args:
        focal_weight: Weight for focal loss component (default: 0.5)
        dice_weight: Weight for dice loss component (default: 0.5)
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        dice_smooth: Smooth parameter for dice loss
    """
    
    def __init__(self, focal_weight=0.5, dice_weight=0.5, 
                 focal_alpha=0.25, focal_gamma=2.0, dice_smooth=1e-6):
        super(ComboLoss, self).__init__()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, 1, H, W)
            target: Ground truth (B, 1, H, W)
        
        Returns:
            loss: Weighted combination of focal and dice loss
        """
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        combo = self.focal_weight * focal + self.dice_weight * dice
        
        return combo


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Dice Loss
    Alternative to ComboLoss (simpler, sometimes works better)
    
    Args:
        bce_weight: Weight for BCE
        dice_weight: Weight for Dice
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCELoss()
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        return self.bce_weight * bce + self.dice_weight * dice


class LogCoshDiceLoss(nn.Module):
    """
    Log-Cosh Dice Loss - Smooth and robust variant of Dice Loss
    
    Uses log(cosh(x)) as a smooth approximation that:
    - Behaves like x^2 for small values (smooth gradients)
    - Behaves like |x| for large values (less sensitive to outliers)
    
    Formula: LogCoshDice = log(cosh(DiceLoss))
             where DiceLoss = 1 - DiceScore
    
    Benefits:
    - Smoother gradients than standard Dice Loss
    - More robust to outliers than MSE
    - Helps with training stability
    - Works well with medical image segmentation
    
    Args:
        smooth: Smoothing factor for Dice calculation
    """
    
    def __init__(self, smooth=1e-6):
        super(LogCoshDiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, 1, H, W) with values in [0, 1]
            target: Ground truth (B, 1, H, W) with values in {0, 1}
        
        Returns:
            loss: Scalar log-cosh dice loss value
        """
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        # Calculate Dice score
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss = 1 - Dice score
        dice_loss = 1.0 - dice_score
        
        # Apply log(cosh(x)) transformation
        # ⚠️ NUMERICAL STABILITY: cosh(x) grows exponentially
        #    - cosh(10) ≈ 11,000
        #    - cosh(50) ≈ 2.6e21  
        #    - cosh(88) = inf → log(inf) = NaN
        # Clamp dice_loss to prevent overflow
        dice_loss_clamped = torch.clamp(dice_loss, min=-50.0, max=50.0)
        
        # log(cosh(x)) is numerically stable and smooth for reasonable x
        logcosh_loss = torch.log(torch.cosh(dice_loss_clamped))
        
        return logcosh_loss


class ComboLogCoshDiceLoss(nn.Module):
    """
    Combination of Focal Loss and Log-Cosh Dice Loss
    
    Enhanced version of ComboLoss that uses Log-Cosh Dice for better stability
    
    Args:
        focal_weight: Weight for focal loss component (default: 0.5)
        dice_weight: Weight for log-cosh dice loss component (default: 0.5)
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        dice_smooth: Smooth parameter for dice loss
    """
    
    def __init__(self, focal_weight=0.5, dice_weight=0.5, 
                 focal_alpha=0.25, focal_gamma=2.0, dice_smooth=1e-6):
        super(ComboLogCoshDiceLoss, self).__init__()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.logcosh_dice_loss = LogCoshDiceLoss(smooth=dice_smooth)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, 1, H, W)
            target: Ground truth (B, 1, H, W)
        
        Returns:
            loss: Weighted combination of focal and log-cosh dice loss
        """
        focal = self.focal_loss(pred, target)
        logcosh_dice = self.logcosh_dice_loss(pred, target)
        
        # Safety check: Replace NaN/Inf with large but finite values
        if torch.isnan(focal) or torch.isinf(focal):
            focal = torch.tensor(10.0, device=focal.device, dtype=focal.dtype)
        if torch.isnan(logcosh_dice) or torch.isinf(logcosh_dice):
            logcosh_dice = torch.tensor(10.0, device=logcosh_dice.device, dtype=logcosh_dice.dtype)
        
        combo = self.focal_weight * focal + self.dice_weight * logcosh_dice
        
        # Final safety: Clamp combined loss to reasonable range
        combo = torch.clamp(combo, min=0.0, max=100.0)
        
        return combo


class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss Wrapper
    
    Wraps any base loss function to handle deep supervision outputs
    from models like AttentionUNetDeepSupervision
    
    Deep supervision provides gradient signals at multiple scales:
    - Main output: Full resolution, weight = 1.0
    - Auxiliary outputs: Intermediate resolutions, decreasing weights
    
    Benefits:
    - Better gradient flow to early layers
    - Faster convergence
    - Improved final performance
    - Multi-scale feature learning
    
    Args:
        base_loss: The base loss function (e.g., DiceLoss, ComboLoss)
        weights: List of weights for [main, aux1, aux2, aux3, ...]
                Default: [1.0, 0.5, 0.25, 0.125] - exponentially decreasing
        num_aux_outputs: Number of auxiliary outputs (default: 3)
    
    Example:
        >>> base_loss = ComboLogCoshDiceLoss()
        >>> ds_loss = DeepSupervisionLoss(base_loss, num_aux_outputs=3)
        >>> # During training:
        >>> outputs = model(x, return_aux=True)  # [main, aux1, aux2, aux3]
        >>> loss = ds_loss(outputs, target)
    """
    
    def __init__(self, base_loss, weights=None, num_aux_outputs=3):
        super(DeepSupervisionLoss, self).__init__()
        
        self.base_loss = base_loss
        self.num_aux_outputs = num_aux_outputs
        
        # Default weights: exponentially decreasing
        if weights is None:
            # Main output: 1.0, Aux outputs: 0.5, 0.25, 0.125, ...
            weights = [1.0] + [0.5 ** (i + 1) for i in range(num_aux_outputs)]
        
        self.weights = weights
        
        # Normalize weights so they sum to a reasonable value
        # This prevents loss magnitude from changing drastically
        total_weight = sum(self.weights)
        self.normalized_weights = [w / total_weight for w in self.weights]
        
        print(f"\n🔥 Deep Supervision Loss Initialized:")
        print(f"   Base Loss: {type(base_loss).__name__}")
        print(f"   Num Aux Outputs: {num_aux_outputs}")
        print(f"   Raw Weights: {self.weights}")
        print(f"   Normalized Weights: {[f'{w:.3f}' for w in self.normalized_weights]}")
    
    def forward(self, outputs, target):
        """
        Args:
            outputs: List of [main_output, aux_output1, aux_output2, ...]
                    Each tensor is (B, 1, H, W)
                    OR single tensor (B, 1, H, W) if no deep supervision
            target: Ground truth (B, 1, H, W)
        
        Returns:
            loss: Weighted combination of losses from all outputs
        """
        # Handle case where model doesn't use deep supervision
        if not isinstance(outputs, list):
            # Single output - just use base loss
            return self.base_loss(outputs, target)
        
        # Deep supervision: compute loss for each output
        total_loss = 0.0
        
        # Ensure target has channel dimension (N, C, H, W)
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)
        
        for i, output in enumerate(outputs):
            # Get weight for this output
            if i < len(self.normalized_weights):
                weight = self.normalized_weights[i]
            else:
                weight = 0.0  # Ignore extra outputs
            
            if weight > 0:
                # Ensure output matches target size
                if output.shape != target.shape:
                    # Upsample output to match target spatial size
                    target_size = target.shape[2:]  # (H, W)
                    output = F.interpolate(output, size=target_size, 
                                         mode='bilinear', align_corners=True)
                
                # Compute loss for this output
                loss_i = self.base_loss(output, target)
                
                # Add weighted loss
                total_loss += weight * loss_i
        
        return total_loss


# ==================== Loss Factory ====================

def get_loss_function(loss_type='combo', **kwargs):
    """
    Factory function to get loss function based on type
    
    Args:
        loss_type: 'focal', 'dice', 'combo', 'tversky', 'bce_dice', 
                   'logcosh_dice', or 'combo_logcosh_dice'
        **kwargs: Additional parameters for the loss function
    
    Returns:
        loss_fn: Loss function instance
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('focal_alpha', 0.25),
            gamma=kwargs.get('focal_gamma', 2.0)
        )
    
    elif loss_type == 'dice':
        return DiceLoss(
            smooth=kwargs.get('dice_smooth', 1e-6)
        )
    
    elif loss_type == 'combo':
        return ComboLoss(
            focal_weight=kwargs.get('focal_weight', 0.5),
            dice_weight=kwargs.get('dice_weight', 0.5),
            focal_alpha=kwargs.get('focal_alpha', 0.25),
            focal_gamma=kwargs.get('focal_gamma', 2.0),
            dice_smooth=kwargs.get('dice_smooth', 1e-6)
        )
    
    elif loss_type == 'logcosh_dice':
        return LogCoshDiceLoss(
            smooth=kwargs.get('dice_smooth', 1e-6)
        )
    
    elif loss_type == 'combo_logcosh_dice':
        return ComboLogCoshDiceLoss(
            focal_weight=kwargs.get('focal_weight', 0.5),
            dice_weight=kwargs.get('dice_weight', 0.5),
            focal_alpha=kwargs.get('focal_alpha', 0.25),
            focal_gamma=kwargs.get('focal_gamma', 2.0),
            dice_smooth=kwargs.get('dice_smooth', 1e-6)
        )
    
    elif loss_type == 'tversky':
        return TverskyLoss(
            alpha=kwargs.get('tversky_alpha', 0.5),
            beta=kwargs.get('tversky_beta', 0.5),
            smooth=kwargs.get('smooth', 1e-6)
        )
    
    elif loss_type == 'bce_dice':
        return BCEDiceLoss(
            bce_weight=kwargs.get('bce_weight', 0.5),
            dice_weight=kwargs.get('dice_weight', 0.5),
            smooth=kwargs.get('smooth', 1e-6)
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ==================== Testing Functions ====================

def test_losses():
    """ทดสอบ loss functions ทั้งหมด"""
    print("🧪 Testing Loss Functions...\n")
    
    # Create dummy data
    batch_size = 4
    height, width = 256, 256
    
    pred = torch.rand(batch_size, 1, height, width)  # Random predictions [0, 1]
    target = (torch.rand(batch_size, 1, height, width) > 0.5).float()  # Binary targets
    
    print(f"Input shapes: pred={pred.shape}, target={target.shape}")
    print(f"Pred range: [{pred.min():.3f}, {pred.max():.3f}]")
    print(f"Target unique values: {target.unique()}\n")
    
    # Test each loss
    losses = {
        'Focal Loss': FocalLoss(),
        'Dice Loss': DiceLoss(),
        'Log-Cosh Dice Loss': LogCoshDiceLoss(),
        'Tversky Loss': TverskyLoss(),
        'Combo Loss': ComboLoss(),
        'Combo LogCosh Dice Loss': ComboLogCoshDiceLoss(),
        'BCE+Dice Loss': BCEDiceLoss()
    }
    
    print("="*50)
    print("Loss Function Results:")
    print("="*50)
    
    for name, loss_fn in losses.items():
        loss_value = loss_fn(pred, target)
        print(f"{name:20s}: {loss_value.item():.6f}")
    
    # Test loss factory
    print("\n" + "="*50)
    print("Testing Loss Factory:")
    print("="*50)
    
    factory_losses = ['focal', 'dice', 'combo', 'logcosh_dice', 'combo_logcosh_dice', 'tversky', 'bce_dice']
    
    for loss_type in factory_losses:
        loss_fn = get_loss_function(loss_type)
        loss_value = loss_fn(pred, target)
        print(f"get_loss_function('{loss_type}'):  {loss_value.item():.6f}")
    
    # Test gradient flow
    print("\n" + "="*50)
    print("Testing Gradient Flow:")
    print("="*50)
    
    pred_with_grad = torch.rand(2, 1, 64, 64, requires_grad=True)
    target_test = (torch.rand(2, 1, 64, 64) > 0.5).float()
    
    combo_loss = ComboLoss()
    loss = combo_loss(pred_with_grad, target_test)
    loss.backward()
    
    print(f"Loss value: {loss.item():.6f}")
    print(f"Gradient computed: {pred_with_grad.grad is not None}")
    print(f"Gradient shape: {pred_with_grad.grad.shape}")
    print(f"Gradient mean: {pred_with_grad.grad.mean().item():.6f}")
    
    print("\n✅ All loss function tests passed!")


if __name__ == "__main__":
    test_losses()
