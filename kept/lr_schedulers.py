"""
Advanced Learning Rate Schedulers for Deep Learning Training
=============================================================

This module implements 6 advanced learning rate scheduling strategies to improve
training stability, convergence speed, and final model performance.

WHY ADVANCED SCHEDULERS MATTER:
-------------------------------
With attention mechanisms (SE, CBAM, ECA, etc.), training can become unstable because:
1. Attention amplifies gradients through multiplicative gating
2. Early training with large LR can cause gradient explosion → NaN loss
3. Simple schedulers may not adapt to training dynamics

IMPLEMENTED SCHEDULERS:
-----------------------

1. WARMUP_COSINE (Recommended for Attention Models)
   - Combines warmup + cosine annealing
   - Warmup phase: 0 → initial_lr (prevents early instability)
   - Main phase: Cosine decay to min_lr
   - Use when: Training with attention, large batch size, or getting NaN early
   - Example: SCHEDULER='warmup_cosine', WARMUP_EPOCHS=5

2. COSINE_RESTARTS (For Escaping Local Minima)
   - SGDR with warm restarts (Loshchilov & Hutter, 2017)
   - Periodically resets LR to escape bad local minima
   - Cycles can grow longer with CYCLE_MULT
   - Use when: Stuck at local minimum, want better generalization
   - Example: SCHEDULER='cosine_restarts', FIRST_CYCLE_EPOCHS=50

3. ONECYCLE (For Fast Training)
   - Super-convergence with 3 phases: warmup → anneal → fine-tune
   - Can achieve same results in fewer epochs
   - Requires careful max_lr tuning
   - Use when: Want faster training, have time for experimentation
   - Example: SCHEDULER='onecycle', LEARNING_RATE=0.01 (higher than normal)

4. ADAPTIVE (For Automatic Handling)
   - Automatically reduces LR when loss increases or plateaus
   - Can detect and recover from NaN
   - No manual tuning needed
   - Use when: Want hands-off training, dealing with unstable loss
   - Example: SCHEDULER='adaptive'

5. POLYNOMIAL (For Smooth Decay)
   - Smooth polynomial decay: lr = initial_lr * (1 - epoch/max_epochs)^power
   - Gentler than exponential, less aggressive than cosine
   - Use when: Fine-tuning, want predictable decay
   - Example: SCHEDULER='polynomial', POLY_POWER=2.0

6. EXPONENTIAL (Classic Approach)
   - Standard exponential decay: lr = initial_lr * gamma^epoch
   - Simple and predictable
   - Use when: Baseline comparison, simple problems
   - Example: SCHEDULER='exponential', EXP_GAMMA=0.95

USAGE IN CONFIG.PY:
-------------------
SCHEDULER = 'warmup_cosine'  # Choose from above
WARMUP_EPOCHS = 5            # For warmup-based schedulers
FIRST_CYCLE_EPOCHS = 50      # For cosine_restarts
CYCLE_MULT = 1               # Cycle length multiplier
POLY_POWER = 2.0             # Polynomial power
EXP_GAMMA = 0.95             # Exponential decay rate

TRAINING STABILITY TIPS:
------------------------
1. Always use warmup with attention mechanisms (WARMUP_EPOCHS=5)
2. If you get NaN loss, try 'adaptive' scheduler
3. If training plateaus, try 'cosine_restarts'
4. For fastest training, try 'onecycle' with higher LR
5. Combine with gradient clipping (GRADIENT_CLIP_VALUE=0.5)

INTEGRATION:
------------
from lr_schedulers import get_advanced_scheduler

scheduler, metadata = get_advanced_scheduler(
    optimizer=optimizer,
    scheduler_type='warmup_cosine',
    config=cfg
)

# In training loop:
if epoch < cfg.WARMUP_EPOCHS and warmup_scheduler:
    warmup_scheduler.step(epoch)
else:
    scheduler.step()
"""

import torch
import torch.optim as optim
import math
import numpy as np
from typing import Optional


class WarmupScheduler:
    """
    Learning Rate Warmup
    Gradually increases LR from 0 to initial_lr over warmup_epochs
    
    Benefits:
    - Prevents early training instability
    - Helps with batch normalization layers
    - Critical for large batch sizes and attention mechanisms
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup
        initial_lr: Target learning rate after warmup
        warmup_method: 'linear' or 'exponential'
    """
    
    def __init__(self, optimizer, warmup_epochs, initial_lr, warmup_method='linear'):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.warmup_method = warmup_method
        self.current_epoch = 0
        
        # Store original LR for each param group
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    
    def step(self, epoch=None):
        """Update learning rate"""
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.current_epoch < self.warmup_epochs:
            if self.warmup_method == 'linear':
                # Linear warmup
                lr_scale = (self.current_epoch + 1) / self.warmup_epochs
            else:
                # Exponential warmup
                lr_scale = math.exp(
                    math.log(self.current_epoch + 1) / self.warmup_epochs * math.log(self.warmup_epochs)
                ) / self.warmup_epochs
            
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * lr_scale
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        """Get current learning rates"""
        return [pg['lr'] for pg in self.optimizer.param_groups]


class CosineAnnealingWarmupRestarts:
    """
    Cosine Annealing with Warm Restarts (SGDR)
    
    Benefits:
    - Escapes local minima through restarts
    - Explores more of loss landscape
    - Often finds better solutions than standard cosine
    
    Paper: "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov & Hutter, 2016)
    
    Args:
        optimizer: PyTorch optimizer
        first_cycle_epochs: Length of first cycle
        cycle_mult: Factor to multiply cycle length after each restart (default: 1)
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        warmup_epochs: Warmup epochs at start of each cycle
        gamma: LR reduction factor after each restart (default: 1.0)
    """
    
    def __init__(self, optimizer, first_cycle_epochs=50, cycle_mult=1, 
                 max_lr=1e-3, min_lr=1e-6, warmup_epochs=5, gamma=1.0):
        self.optimizer = optimizer
        self.first_cycle_epochs = first_cycle_epochs
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.gamma = gamma
        
        self.current_epoch = 0
        self.cycle = 0
        self.cycle_epoch = 0
        self.current_cycle_epochs = first_cycle_epochs
    
    def step(self):
        """Update learning rate"""
        self.current_epoch += 1
        self.cycle_epoch += 1
        
        # Check if we need to restart
        if self.cycle_epoch >= self.current_cycle_epochs:
            self.cycle += 1
            self.cycle_epoch = 0
            self.current_cycle_epochs = int(self.first_cycle_epochs * (self.cycle_mult ** self.cycle))
            self.max_lr *= self.gamma
        
        # Calculate LR
        if self.cycle_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.min_lr + (self.max_lr - self.min_lr) * (self.cycle_epoch / self.warmup_epochs)
        else:
            # Cosine annealing phase
            progress = (self.cycle_epoch - self.warmup_epochs) / (self.current_cycle_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


class OneCycleLR:
    """
    One Cycle Learning Rate Policy
    
    Benefits:
    - Fast convergence (fewer epochs needed)
    - Better generalization
    - Automatic LR range finding
    
    Paper: "Super-Convergence" (Smith, 2018)
    
    Strategy:
    1. Phase 1 (0-30%): LR increases from min to max
    2. Phase 2 (30-90%): LR decreases from max to min
    3. Phase 3 (90-100%): LR decreases further (annihilation)
    
    Args:
        optimizer: PyTorch optimizer
        max_lr: Maximum learning rate
        total_steps: Total number of training steps
        pct_start: Percentage of steps for phase 1 (default: 0.3)
        anneal_strategy: 'cos' or 'linear'
        div_factor: Initial LR = max_lr / div_factor (default: 25)
        final_div_factor: Final LR = max_lr / final_div_factor (default: 1e4)
    """
    
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, 
                 anneal_strategy='cos', div_factor=25.0, final_div_factor=1e4):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        self.step_count = 0
        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up
    
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        
        if self.step_count <= self.step_size_up:
            # Phase 1: Increase LR
            progress = self.step_count / self.step_size_up
            if self.anneal_strategy == 'cos':
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * (1 - math.cos(math.pi * progress)) / 2
            else:
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Phase 2 & 3: Decrease LR
            progress = (self.step_count - self.step_size_up) / self.step_size_down
            if self.anneal_strategy == 'cos':
                lr = self.final_lr + (self.max_lr - self.final_lr) * (1 + math.cos(math.pi * progress)) / 2
            else:
                lr = self.max_lr - (self.max_lr - self.final_lr) * progress
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


class PolynomialLR:
    """
    Polynomial Learning Rate Decay
    
    Benefits:
    - Smooth decay
    - Configurable decay rate
    - Good for fine-tuning
    
    Args:
        optimizer: PyTorch optimizer
        total_epochs: Total number of epochs
        power: Polynomial power (default: 2.0)
        min_lr: Minimum learning rate
    """
    
    def __init__(self, optimizer, total_epochs, power=2.0, min_lr=1e-7):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate"""
        self.current_epoch += 1
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            decay = (1 - self.current_epoch / self.total_epochs) ** self.power
            lr = max(self.base_lrs[i] * decay, self.min_lr)
            param_group['lr'] = lr
    
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


class AdaptiveScheduler:
    """
    Adaptive Learning Rate Scheduler with Loss Monitoring
    
    Benefits:
    - Automatically adjusts LR based on training dynamics
    - Detects plateaus and convergence
    - Prevents overfitting
    
    Strategy:
    - If loss doesn't improve for patience epochs → reduce LR
    - If loss increases significantly → reduce LR immediately
    - If training is unstable (NaN/Inf) → reduce LR aggressively
    
    Args:
        optimizer: PyTorch optimizer
        mode: 'min' or 'max' (monitoring loss or accuracy)
        factor: LR reduction factor (default: 0.5)
        patience: Epochs to wait before reducing (default: 10)
        threshold: Minimum change to qualify as improvement (default: 1e-4)
        min_lr: Minimum learning rate
        cooldown: Epochs to wait after LR reduction (default: 0)
        verbose: Print LR changes
    """
    
    def __init__(self, optimizer, mode='min', factor=0.5, patience=10, 
                 threshold=1e-4, min_lr=1e-7, cooldown=0, verbose=True):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.cooldown = cooldown
        self.verbose = verbose
        
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.last_epoch = 0
        
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
    
    def step(self, metric):
        """
        Update learning rate based on metric
        
        Args:
            metric: Current metric value (loss or accuracy)
        """
        if math.isnan(metric) or math.isinf(metric):
            # Handle NaN/Inf - reduce LR aggressively
            self._reduce_lr(factor=0.1)
            if self.verbose:
                print(f"⚠️  NaN/Inf detected! Reducing LR by factor 0.1")
            return
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
            return
        
        if self.best is None:
            self.best = metric
            return
        
        # Check if metric improved
        if self.mode == 'min':
            improved = metric < (self.best - self.threshold)
        else:
            improved = metric > (self.best + self.threshold)
        
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        # Reduce LR if no improvement for patience epochs
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
    
    def _reduce_lr(self, factor=None):
        """Reduce learning rate"""
        if factor is None:
            factor = self.factor
        
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if self.verbose and old_lr != new_lr:
                print(f"   📉 Reducing LR: {old_lr:.2e} → {new_lr:.2e}")
    
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


def get_advanced_scheduler(optimizer, scheduler_type, config):
    """
    Factory function to create advanced schedulers
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        config: Configuration object
    
    Returns:
        scheduler instance and metadata
    """
    
    total_epochs = config.NUM_EPOCHS
    initial_lr = config.LEARNING_RATE
    min_lr = getattr(config, 'SCHEDULER_MIN_LR', 1e-7)
    
    metadata = {'type': scheduler_type, 'requires_metric': False}
    
    if scheduler_type == 'warmup_cosine':
        # Warmup + Cosine Annealing
        warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 5)
        
        # Create warmup scheduler
        warmup = WarmupScheduler(optimizer, warmup_epochs, initial_lr, 'linear')
        
        # Create cosine scheduler
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr
        )
        
        metadata['warmup_epochs'] = warmup_epochs
        metadata['description'] = f'Linear Warmup ({warmup_epochs} epochs) + Cosine Annealing'
        
        return {'warmup': warmup, 'main': cosine}, metadata
    
    elif scheduler_type == 'cosine_restarts':
        # Cosine Annealing with Warm Restarts
        first_cycle = getattr(config, 'FIRST_CYCLE_EPOCHS', 50)
        cycle_mult = getattr(config, 'CYCLE_MULT', 1)
        
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_epochs=first_cycle,
            cycle_mult=cycle_mult,
            max_lr=initial_lr,
            min_lr=min_lr,
            warmup_epochs=5,
            gamma=0.9
        )
        
        metadata['description'] = f'Cosine Annealing with Warm Restarts (cycle={first_cycle})'
        
        return scheduler, metadata
    
    elif scheduler_type == 'one_cycle':
        # One Cycle LR
        train_loader_size = getattr(config, '_train_loader_size', 100)
        total_steps = total_epochs * train_loader_size
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=initial_lr * 10,  # Peak LR is 10x initial
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        metadata['description'] = 'One Cycle LR (Super-Convergence)'
        metadata['step_per_batch'] = True  # Needs to step every batch
        
        return scheduler, metadata
    
    elif scheduler_type == 'polynomial':
        # Polynomial LR Decay
        power = getattr(config, 'POLY_POWER', 2.0)
        
        scheduler = PolynomialLR(
            optimizer,
            total_epochs=total_epochs,
            power=power,
            min_lr=min_lr
        )
        
        metadata['description'] = f'Polynomial Decay (power={power})'
        
        return scheduler, metadata
    
    elif scheduler_type == 'adaptive':
        # Adaptive Scheduler
        patience = getattr(config, 'SCHEDULER_PATIENCE', 10)
        factor = getattr(config, 'SCHEDULER_FACTOR', 0.5)
        
        scheduler = AdaptiveScheduler(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            threshold=1e-4,
            min_lr=min_lr,
            cooldown=5,
            verbose=True
        )
        
        metadata['description'] = f'Adaptive LR (patience={patience}, factor={factor})'
        metadata['requires_metric'] = True
        
        return scheduler, metadata
    
    elif scheduler_type == 'exponential':
        # Exponential LR Decay
        gamma = getattr(config, 'EXP_GAMMA', 0.95)
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )
        
        metadata['description'] = f'Exponential Decay (gamma={gamma})'
        
        return scheduler, metadata
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ============================================================================
# SCHEDULER RECOMMENDATIONS
# ============================================================================

SCHEDULER_GUIDE = """
🎯 LEARNING RATE SCHEDULER GUIDE

## 1. Warmup + Cosine ('warmup_cosine') ⭐ RECOMMENDED FOR ATTENTION
   - Best for: Attention mechanisms, transformers, large models
   - Benefits: Stable start, smooth decay
   - Settings: WARMUP_EPOCHS = 5-10
   - Use when: Training with attention modules, large batch size

## 2. Cosine with Restarts ('cosine_restarts') ⭐ BEST FOR EXPLORATION
   - Best for: Finding better local minima
   - Benefits: Escapes local minima, explores loss landscape
   - Settings: FIRST_CYCLE_EPOCHS = 50, CYCLE_MULT = 1 or 2
   - Use when: Want to explore different solutions

## 3. One Cycle LR ('one_cycle') ⭐ FASTEST CONVERGENCE
   - Best for: Fast training, fewer epochs
   - Benefits: Super-convergence, better generalization
   - Settings: Automatic
   - Use when: Time is limited, need fast results

## 4. Polynomial Decay ('polynomial')
   - Best for: Fine-tuning, transfer learning
   - Benefits: Smooth, controlled decay
   - Settings: POLY_POWER = 1.0-3.0
   - Use when: Fine-tuning pre-trained models

## 5. Adaptive ('adaptive') ⭐ MOST ROBUST
   - Best for: Unstable training, NaN issues
   - Benefits: Automatic adjustment, handles instability
   - Settings: SCHEDULER_PATIENCE = 10, SCHEDULER_FACTOR = 0.5
   - Use when: Training is unstable, frequent NaN

## 6. Exponential ('exponential')
   - Best for: Long training runs
   - Benefits: Simple, consistent decay
   - Settings: EXP_GAMMA = 0.90-0.99
   - Use when: Standard training, no special requirements

## RECOMMENDATIONS:

### For Medical Image Segmentation with Attention:
   USE: 'warmup_cosine' or 'adaptive'
   - Warmup prevents early instability
   - Cosine provides smooth learning

### For Fast Experimentation:
   USE: 'one_cycle'
   - Train in half the epochs
   - Still get good results

### For Maximum Performance:
   USE: 'cosine_restarts'
   - Finds better solutions
   - May take longer

### For Stability (NaN Issues):
   USE: 'adaptive'
   - Automatically handles issues
   - Safe choice for attention
"""


if __name__ == "__main__":
    print(SCHEDULER_GUIDE)
