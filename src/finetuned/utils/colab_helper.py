"""Utility functions untuk Google Colab environment."""

import torch
from typing import Dict, Any


class ColabHelper:
    """Helper class untuk setup dan manage Google Colab environment."""
    
    @staticmethod
    def check_gpu() -> Dict[str, Any]:
        """
        Check GPU availability dan specifications.
        
        Returns:
            Dict dengan:
            - gpu_available: bool
            - gpu_name: str
            - gpu_memory_gb: float
            
        Raises:
            RuntimeError: Jika GPU tidak available
        """
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available! Please enable GPU in Colab Runtime settings.")
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        result = {
            "gpu_available": True,
            "gpu_name": gpu_name,
            "gpu_memory_gb": round(gpu_memory, 2)
        }
        
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {result['gpu_memory_gb']:.2f} GB")
        
        return result
    
    @staticmethod
    def mount_drive() -> bool:
        """
        Mount Google Drive untuk persistent storage.
        
        Returns:
            bool: True jika berhasil mount
        """
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✓ Google Drive mounted successfully")
            return True
        except ImportError:
            print("⚠ Not running in Colab environment, skipping Drive mount")
            return False
        except Exception as e:
            print(f"✗ Failed to mount Google Drive: {e}")
            return False
    
    @staticmethod
    def install_dependencies() -> None:
        """
        Install required packages untuk fine-tuning.
        
        Packages:
        - transformers: HuggingFace Transformers
        - peft: Parameter-Efficient Fine-Tuning
        - datasets: HuggingFace Datasets
        - accelerate: Training acceleration
        - bitsandbytes: Quantization support
        - evaluate: Evaluation metrics
        - rouge_score: ROUGE metrics
        - bert_score: BERTScore
        """
        import subprocess
        import sys
        
        packages = [
            "transformers>=4.35.0",
            "peft>=0.7.0",
            "datasets>=2.15.0",
            "accelerate>=0.25.0",
            "bitsandbytes>=0.41.0",
            "evaluate>=0.4.0",
            "rouge_score>=0.1.2",
            "bert_score>=0.3.13",
        ]
        
        print("Installing dependencies...")
        for package in packages:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        
        print("✓ All dependencies installed successfully")
    
    @staticmethod
    def setup_wandb(project_name: str = "indot5-aqg", api_key: str = None) -> None:
        """
        Setup Weights & Biases untuk experiment tracking (optional).
        
        Args:
            project_name: Nama project di W&B
            api_key: W&B API key (optional, akan prompt jika tidak provided)
        """
        try:
            import wandb
            
            if api_key:
                wandb.login(key=api_key)
            else:
                wandb.login()
            
            print(f"✓ W&B initialized for project: {project_name}")
        except ImportError:
            print("⚠ wandb not installed. Install with: pip install wandb")
        except Exception as e:
            print(f"✗ Failed to setup W&B: {e}")
