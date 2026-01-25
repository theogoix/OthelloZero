# %%
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Union
import json
from datetime import datetime
import argparse

from config import TrainingConfig
from data_loader import (
    OthelloDataLoader,
    OthelloDataset,
    load_multiple_files,
    print_dataset_statistics
)
from model import OthelloNet

'''
class TrainingConfig:
    """Configuration for training"""
    def __init__(self):
        # Data
        self.data_paths = ["../../data/first_iteration.bin"]  # Can be list of files
        self.train_split = 0.9  # 90% train, 10% validation
        
        # Model
        self.model_size = 'small'  # 'tiny', 'small', 'medium', 'large'
        self.num_filters = None  # Override model_size if specified
        self.num_res_blocks = None
        
        # Training
        self.batch_size = 256
        self.num_epochs = 10
        self.learning_rate = 0.001
        self.weight_decay = 1e-4

        
        # Augmentation
        self.use_augmentation = True
        self.use_mix_z_q = True
        
        # Fine-tuning
        self.pretrained_path = None  # Path to checkpoint to continue from
        self.reset_optimizer = False  # If True, don't load optimizer state
        
        # Hardware
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 4
        
        # Checkpointing
        self.checkpoint_dir = "../../models/checkpoints"
        self.save_every = 1  # Save every N epochs
        
        # Logging
        self.log_every = 100  # Log every N batches
        self.verbose = True
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, path: str):
        """Save configuration to JSON"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON"""
        config = cls()
        with open(path, 'r') as f:
            data = json.load(f)
            for k, v in data.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        return config
'''

class Trainer:
    """Trainer for Othello neural network"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler (reduce LR when plateauing)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=5,
            gamma=0.5
        )
        
        # Track best model
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        for batch_idx, (states, policy_targets, value_targets) in enumerate(train_loader):
            # Move to device
            states = states.to(self.device)
            policy_targets = policy_targets.to(self.device)
            value_targets = value_targets.to(self.device)
            
            # Forward pass
            policy_pred, value_pred = self.model(states)
            
            # Compute loss
            loss, policy_loss, value_loss = self.compute_loss(
                policy_pred, value_pred,
                policy_targets, value_targets
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
            
            # Log progress
            if self.config.verbose and (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = total_loss / num_batches
                avg_policy = total_policy_loss / num_batches
                avg_value = total_value_loss / num_batches
                
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                      f"Loss={avg_loss:.4f} (Policy={avg_policy:.4f}, Value={avg_value:.4f})")
        
        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for states, policy_targets, value_targets in val_loader:
                # Move to device
                states = states.to(self.device)
                policy_targets = policy_targets.to(self.device)
                value_targets = value_targets.to(self.device)
                
                # Forward pass
                policy_pred, value_pred = self.model(states)
                
                # Compute loss
                loss, policy_loss, value_loss = self.compute_loss(
                    policy_pred, value_pred,
                    policy_targets, value_targets
                )
                
                # Track statistics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
        }
    
    def compute_loss(
        self,
        policy_pred: torch.Tensor,
        value_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor
    ) -> tuple:
        """Compute combined loss (policy + value)"""
        # Policy loss: cross-entropy
        # policy_pred is already softmax'd, policy_target is a distribution
        policy_loss = -torch.sum(policy_target * torch.log(policy_pred + 1e-8), dim=1).mean()
        
        # Value loss: MSE
        value_loss = nn.functional.mse_loss(value_pred.squeeze(), value_target.squeeze())
        
        # Combined loss (equal weighting)
        total_loss = policy_loss + value_loss
        
        return total_loss, policy_loss, value_loss
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config.to_dict(),
            'model_config': self.model.get_config(),
        }
        
        # Save regular checkpoint
        path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        if self.config.verbose:
            print(f"  Saved checkpoint: {path}")
        
        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            if self.config.verbose:
                print(f"  ✓ New best model saved: {best_path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load model checkpoint for fine-tuning"""
        if self.config.verbose:
            print(f"Loading checkpoint from: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Optionally load optimizer state (for continuing training)
        if load_optimizer and not self.config.reset_optimizer:
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.config.verbose:
                print("  Loaded optimizer and scheduler state")
        else:
            if self.config.verbose:
                print("  Starting with fresh optimizer")
        
        # Load training state
        self.start_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.config.verbose:
            print(f"  Resuming from epoch {self.start_epoch}")
            print(f"  Previous best val loss: {self.best_val_loss:.4f}")
        
        return checkpoint


def train(config: TrainingConfig):
    """Main training function"""
    if config.verbose:
        print("=" * 70)
        print("Othello Neural Network Training")
        print("=" * 70)
        print(f"\nConfiguration:")
        for k, v in config.to_dict().items():
            print(f"  {k}: {v}")
        print()
    
    # Load data (can handle multiple files)
    if config.verbose:
        print("Loading training data...")
    examples = load_multiple_files(config.data_paths, verbose=config.verbose)
    
    if not examples:
        raise ValueError("No training examples loaded!")
    
    print_dataset_statistics(examples)
    
    # Split into train/validation
    np.random.shuffle(examples)  # Shuffle before split
    train_size = int(len(examples) * config.train_split)
    
    train_examples = examples[:train_size]
    val_examples = examples[train_size:]
    
    if config.verbose:
        print(f"\nTrain/Validation split:")
        print(f"  Train: {len(train_examples):,} examples ({config.train_split:.0%})")
        print(f"  Val:   {len(val_examples):,} examples ({1-config.train_split:.0%})")
    
    # Create datasets
    train_dataset = OthelloDataset(
        train_examples,
        augment=config.use_augmentation,
        use_mix_z_q=config.use_mix_z_q
    )
    
    val_dataset = OthelloDataset(
        val_examples,
        augment=False,  # No augmentation for validation
        use_mix_z_q=config.use_mix_z_q
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False
    )
    
    # Create model
    if config.verbose:
        print(f"\nCreating model...")
    
    if config.num_filters is not None and config.num_res_blocks is not None:
        # Custom size
        model = OthelloNet(config.num_filters, config.num_res_blocks)
        if config.verbose:
            print(f"  Custom model: {config.num_filters} filters, {config.num_res_blocks} blocks")
    else:
        # Predefined size
        model = OthelloNet.from_config(config.model_size)
        if config.verbose:
            print(f"  Model size: {config.model_size}")
            print(f"  Description: {OthelloNet.CONFIGS[config.model_size]['description']}")
    
    num_params = model.count_parameters()
    if config.verbose:
        print(f"  Total parameters: {num_params:,}")
        print(f"  Device: {config.device}")
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Load pretrained model if specified (fine-tuning)
    if config.pretrained_path:
        trainer.load_checkpoint(
            config.pretrained_path,
            load_optimizer=not config.reset_optimizer
        )
    
    
    # Save config
    config.save(Path(config.checkpoint_dir) / "config.json")
    
    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")
    print("="*70)
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print("-" * 70)
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        print("  Validating...")
        val_metrics = trainer.validate(val_loader)
        
        # Update learning rate
        trainer.scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Print summary
        print(f"\n  Epoch {epoch} Summary:")
        print(f"    Train Loss: {train_metrics['loss']:.4f} "
              f"(Policy={train_metrics['policy_loss']:.4f}, "
              f"Value={train_metrics['value_loss']:.4f})")
        print(f"    Val Loss:   {val_metrics['loss']:.4f} "
              f"(Policy={val_metrics['policy_loss']:.4f}, "
              f"Value={val_metrics['value_loss']:.4f})")
        print(f"    Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        is_best = val_metrics['loss'] < trainer.best_val_loss
        if is_best:
            trainer.best_val_loss = val_metrics['loss']
        
        if epoch % config.save_every == 0 or is_best:
            trainer.save_checkpoint(
                epoch,
                {'train': train_metrics, 'val': val_metrics},
                is_best=is_best
            )
    
    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print("="*70)
    
    # Export best model to TorchScript
    print("\nExporting best model to TorchScript...")
    best_checkpoint_path = Path(config.checkpoint_dir) / "best_model.pt"
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scripted_model = torch.jit.script(model)
    export_path = Path(config.checkpoint_dir) / "best_model_scripted.pt"
    scripted_model.save(str(export_path))
    print(f"✓ Exported to: {export_path}")

#%%

if __name__ == "__main__":
    # Create config
    config = TrainingConfig()
    config.pretrained_path = "../../models/checkpoints/best_model.pt"
    config.data_paths = ["../../data/third_iteration.bin", "../../data/fourth_iteration.bin"]
    config.num_epochs = 20
    
    # Override from command line if needed
    if len(sys.argv) > 1:
        config.data_path = sys.argv[1]
    
    # Run training
    train(config)
# %%
