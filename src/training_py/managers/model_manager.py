#%%
"""
Model management for the training pipeline
"""
import shutil
from pathlib import Path
from typing import Optional
import torch


class ModelManager:
    """Manages model versions, promotion, and archiving"""
    
    def __init__(self, models_dir: Path, verbose: bool = True):
        self.models_dir = Path(models_dir)
        self.archived_dir = self.models_dir / "archived"
        self.verbose = verbose
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.archived_dir.mkdir(parents=True, exist_ok=True)
        
        # Track current best model
        self.best_model_path = self.models_dir / "best_model.pt"
        self.best_model_scripted_path = self.models_dir / "best_model_scripted.pt"
    
    def get_best_model(self) -> Optional[Path]:
        """Get path to current best model"""
        if self.best_model_scripted_path.exists():
            return self.best_model_scripted_path
        elif self.best_model_path.exists():
            return self.best_model_path
        else:
            return None
    
    def has_best_model(self) -> bool:
        """Check if a best model exists"""
        return self.get_best_model() is not None
    
    def promote_model(
        self,
        checkpoint_path: Path,
        iteration: int,
        archive_old: bool = True
    ):
        """
        Promote a model to become the new best model
        
        Args:
            checkpoint_path: Path to the checkpoint to promote
            iteration: Iteration number
            archive_old: Whether to archive the old best model
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if self.verbose:
            print(f"\nPromoting model from iteration {iteration}...")
        
        # Archive old best model if it exists
        if archive_old and self.best_model_path.exists():
            self._archive_model(self.best_model_path, "previous_best")
            if self.best_model_scripted_path.exists():
                self._archive_model(self.best_model_scripted_path, "previous_best_scripted")
        
        # Copy checkpoint to best model location
        shutil.copy2(checkpoint_path, self.best_model_path)
        
        # Also copy the scripted version if it exists
        scripted_path = checkpoint_path.parent / "best_model_scripted.pt"
        if scripted_path.exists():
            shutil.copy2(scripted_path, self.best_model_scripted_path)
        
        # Archive the promoted model with iteration number
        iteration_path = self.models_dir / f"model_iter_{iteration}.pt"
        shutil.copy2(checkpoint_path, iteration_path)
        
        if scripted_path.exists():
            iteration_scripted_path = self.models_dir / f"model_iter_{iteration}_scripted.pt"
            shutil.copy2(scripted_path, iteration_scripted_path)
        
        if self.verbose:
            print(f"  ✓ Promoted model to: {self.best_model_path}")
            print(f"  ✓ Archived as: {iteration_path}")
    
    def archive_model(
        self,
        checkpoint_path: Path,
        iteration: int,
        label: str = "rejected"
    ):
        """
        Archive a model (e.g., rejected challenger)
        
        Args:
            checkpoint_path: Path to checkpoint
            iteration: Iteration number
            label: Label for the archive (e.g., "rejected", "checkpoint")
        """
        if not checkpoint_path.exists():
            if self.verbose:
                print(f"Warning: Cannot archive non-existent model: {checkpoint_path}")
            return
        
        archive_name = f"model_iter_{iteration}_{label}.pt"
        archive_path = self.archived_dir / archive_name
        
        shutil.copy2(checkpoint_path, archive_path)
        
        if self.verbose:
            print(f"  Archived model: {archive_path}")
    
    def _archive_model(self, model_path: Path, label: str):
        """Internal method to archive a model"""
        if not model_path.exists():
            return
        
        archive_name = f"{model_path.stem}_{label}{model_path.suffix}"
        archive_path = self.archived_dir / archive_name
        
        shutil.copy2(model_path, archive_path)
        
        if self.verbose:
            print(f"  Archived old best: {archive_path}")
    
    def export_model(
        self,
        checkpoint_path: Path,
        output_path: Path,
        format: str = "torchscript"
    ):
        """
        Export model to different formats
        
        Args:
            checkpoint_path: Path to PyTorch checkpoint
            output_path: Where to save exported model
            format: Export format ('torchscript', 'onnx', 'state_dict')
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if self.verbose:
            print(f"Exporting model to {format}...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if format == "state_dict":
            # Just save the state dict
            torch.save(checkpoint['model_state_dict'], output_path)
            
        elif format == "torchscript":
            # Load full model and export as TorchScript
            # This requires the model class to be available
            from model import OthelloNet
            
            model_config = checkpoint.get('model_config', {})
            model = OthelloNet(
                model_config.get('num_filters', 128),
                model_config.get('num_res_blocks', 10)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            scripted_model = torch.jit.script(model)
            scripted_model.save(str(output_path))
            
        elif format == "onnx":
            # Export to ONNX
            from model import OthelloNet
            import torch.onnx
            
            model_config = checkpoint.get('model_config', {})
            model = OthelloNet(
                model_config.get('num_filters', 128),
                model_config.get('num_res_blocks', 10)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 8, 8)
            
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['policy', 'value'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'policy': {0: 'batch_size'},
                    'value': {0: 'batch_size'}
                }
            )
        
        else:
            raise ValueError(f"Unknown export format: {format}")
        
        if self.verbose:
            print(f"  ✓ Exported to: {output_path}")
    
    def cleanup_old_checkpoints(
        self,
        checkpoint_dir: Path,
        keep_every_n: int = 10,
        keep_last_n: int = 3
    ):
        """
        Clean up old training checkpoints to save space
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            keep_every_n: Keep every Nth checkpoint
            keep_last_n: Keep the last N checkpoints
        """
        if not checkpoint_dir.exists():
            return
        
        # Find all checkpoint files
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(checkpoints) <= keep_last_n:
            return  # Not enough checkpoints to clean
        
        # Determine which to keep
        keep_files = set()
        
        # Keep last N
        for cp in checkpoints[-keep_last_n:]:
            keep_files.add(cp)
        
        # Keep every Nth
        for i, cp in enumerate(checkpoints):
            if (i + 1) % keep_every_n == 0:
                keep_files.add(cp)
        
        # Always keep best_model.pt
        best_path = checkpoint_dir / "best_model.pt"
        if best_path.exists():
            keep_files.add(best_path)
        
        # Delete others
        deleted_count = 0
        for cp in checkpoints:
            if cp not in keep_files:
                cp.unlink()
                deleted_count += 1
        
        if self.verbose and deleted_count > 0:
            print(f"  Cleaned up {deleted_count} old checkpoints")
    
    def get_model_info(self, checkpoint_path: Path) -> dict:
        """Get information about a model checkpoint"""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        return {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'best_val_loss': checkpoint.get('best_val_loss', 'unknown'),
            'model_config': checkpoint.get('model_config', {}),
            'training_config': checkpoint.get('config', {}),
            'metrics': checkpoint.get('metrics', {})
        }
# %%
