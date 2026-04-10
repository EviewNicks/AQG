"""Checkpoint manager untuk save, load, dan cleanup checkpoints."""

import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch


class CheckpointManager:
    """Class untuk manage checkpoints selama training."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 3,
        drive_backup: bool = True,
        drive_path: str = "/content/drive/MyDrive/aqg_checkpoints/"
    ):
        """
        Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory untuk save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            drive_backup: Whether to backup to Google Drive
            drive_path: Path di Google Drive untuk backup
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.drive_backup = drive_backup
        self.drive_path = Path(drive_path)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.drive_backup:
            try:
                self.drive_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"⚠ Warning: Could not create Drive backup directory: {e}")
                self.drive_backup = False
    
    def save_checkpoint(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict[str, float]
    ) -> str:
        """
        Save checkpoint dengan metadata.
        
        Args:
            model: Model to save (PeftModel)
            optimizer: Optimizer state (optional)
            epoch: Current epoch number
            metrics: Training metrics
            
        Returns:
            Path ke saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving checkpoint to {checkpoint_path}...")
        
        # Save model (LoRA adapters only)
        model.save_pretrained(checkpoint_path)
        print("  ✓ Model saved")
        
        # Save optimizer state
        if optimizer is not None:
            optimizer_path = checkpoint_path / "optimizer.pt"
            torch.save(optimizer.state_dict(), optimizer_path)
            print("  ✓ Optimizer state saved")
        
        # Save metadata
        metadata = {
            "epoch": epoch,
            "metrics": metrics
        }
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print("  ✓ Metadata saved")
        
        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()
        
        # Backup to Drive
        if self.drive_backup:
            self.backup_to_drive(str(checkpoint_path))
        
        print(f"✓ Checkpoint saved successfully: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint untuk resume training.
        
        Args:
            checkpoint_path: Path ke checkpoint directory
            
        Returns:
            Dict dengan checkpoint data:
            - epoch: int
            - metrics: Dict[str, float]
            - optimizer_state: Optional[Dict]
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        
        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            print(f"  ✓ Loaded metadata (epoch {metadata['epoch']})")
        else:
            metadata = {"epoch": 0, "metrics": {}}
            print("  ⚠ No metadata found")
        
        # Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            optimizer_state = torch.load(optimizer_path)
            print("  ✓ Loaded optimizer state")
        else:
            optimizer_state = None
            print("  ⚠ No optimizer state found")
        
        result = {
            "epoch": metadata["epoch"],
            "metrics": metadata["metrics"],
            "optimizer_state": optimizer_state,
            "checkpoint_path": str(checkpoint_path)
        }
        
        print(f"✓ Checkpoint loaded successfully")
        return result
    
    def cleanup_old_checkpoints(self) -> None:
        """
        Keep only last N checkpoints.
        """
        # Get all checkpoint directories
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-epoch-")],
            key=lambda x: int(x.name.split("-")[-1])
        )
        
        # Remove old checkpoints
        if len(checkpoints) > self.max_checkpoints:
            to_remove = checkpoints[:-self.max_checkpoints]
            for checkpoint in to_remove:
                print(f"  Removing old checkpoint: {checkpoint.name}")
                shutil.rmtree(checkpoint)
    
    def backup_to_drive(self, checkpoint_path: str) -> None:
        """
        Copy checkpoint ke Google Drive.
        
        Args:
            checkpoint_path: Path ke checkpoint directory
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            backup_path = self.drive_path / checkpoint_path.name
            
            print(f"  Backing up to Drive: {backup_path}")
            
            # Copy directory
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.copytree(checkpoint_path, backup_path)
            
            print(f"  ✓ Backup complete")
        except Exception as e:
            print(f"  ⚠ Backup failed: {e}")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-epoch-")],
            key=lambda x: int(x.name.split("-")[-1])
        )
        
        if checkpoints:
            return str(checkpoints[-1])
        return None
    
    def get_best_checkpoint(self, metric_name: str = "eval_loss", higher_is_better: bool = False) -> Optional[str]:
        """
        Get path to best checkpoint based on metric.
        
        Args:
            metric_name: Name of metric to compare
            higher_is_better: Whether higher metric value is better
            
        Returns:
            Path to best checkpoint or None if no checkpoints exist
        """
        checkpoints = [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-epoch-")]
        
        if not checkpoints:
            return None
        
        best_checkpoint = None
        best_metric = float('-inf') if higher_is_better else float('inf')
        
        for checkpoint in checkpoints:
            metadata_path = checkpoint / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                if metric_name in metadata.get("metrics", {}):
                    metric_value = metadata["metrics"][metric_name]
                    
                    if higher_is_better:
                        if metric_value > best_metric:
                            best_metric = metric_value
                            best_checkpoint = checkpoint
                    else:
                        if metric_value < best_metric:
                            best_metric = metric_value
                            best_checkpoint = checkpoint
        
        if best_checkpoint:
            print(f"Best checkpoint: {best_checkpoint.name} ({metric_name}={best_metric:.4f})")
            return str(best_checkpoint)
        
        return None
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint info dicts
        """
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-epoch-")],
            key=lambda x: int(x.name.split("-")[-1])
        )
        
        checkpoint_info = []
        for checkpoint in checkpoints:
            metadata_path = checkpoint / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            else:
                metadata = {"epoch": 0, "metrics": {}}
            
            info = {
                "path": str(checkpoint),
                "name": checkpoint.name,
                "epoch": metadata["epoch"],
                "metrics": metadata["metrics"]
            }
            checkpoint_info.append(info)
        
        return checkpoint_info
