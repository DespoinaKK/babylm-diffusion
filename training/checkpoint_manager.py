import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List, Tuple
import argparse
import copy

from utils.utils import is_main_process


class CheckpointManager:
    """Handles saving a full training checkpoint and separate inference weights."""
    
    @staticmethod
    def save_checkpoint(model: nn.Module, ema: 'ExponentialMovingAverage',
                    optimizer: torch.optim.Optimizer, scheduler: 'Any',
                    global_step: int, args: 'argparse.Namespace') -> None:
        """
        Saves a full training checkpoint and separate .bin files for
        regular and EMA model weights.
        """
        if not is_main_process():
            return
        
        # 1. Define Output Directory for this step
        output_dir = args.output_path / f"step_{global_step}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Get the underlying model (unwrap DDP if needed)
        model_to_save = model.module if hasattr(model, 'module') else model
        
        # 3. Get the regular model's state_dict
        regular_weights = copy.deepcopy(model_to_save.state_dict())
        
        # 4. Get the EMA model's state_dict using the store/copy/restore method
        # Extract just the parameters (not the names) for EMA methods
        model_parameters = [param for name, param in model_to_save.named_parameters()]
        
        ema.store(model_parameters)  # Pass only parameters, not (name, param) tuples
        ema.copy_to(model_parameters)  # Pass only parameters
        ema_weights = copy.deepcopy(model_to_save.state_dict())
        ema.restore(model_parameters)  # IMPORTANT: restore original weights, pass only parameters
        
        # 5. Save the separate weight files for inference
        torch.save(regular_weights, output_dir / "pytorch_model.bin")
        torch.save(ema_weights, output_dir / "pytorch_model_ema.bin")
        print(f"✅ Saved separate inference weights to: {output_dir}")
        
        # 6. Save the full, comprehensive checkpoint for resuming training
        full_checkpoint_path = output_dir / "training_checkpoint.pt"
        checkpoint_data = {
            "model": regular_weights,  # Use the unwrapped state_dict
            "ema": ema.state_dict(),  # Save the EMA class's internal state
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "args": vars(args)
        }
        torch.save(checkpoint_data, full_checkpoint_path)
        print(f"✅ Saved full training checkpoint to: {full_checkpoint_path}")    
        
    @staticmethod
    def get_save_points(total_tokens: int) -> Tuple[List[int], List[float]]:
        """Calculate token-based checkpoint save points."""
        save_percentages = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 
                           55, 60, 65, 70, 75, 80, 85, 90, 95]
        save_points = [int(total_tokens * pct / 100) for pct in save_percentages]
        return save_points, save_percentages
    
    @staticmethod
    def should_save_checkpoint(tokens_seen: int, save_points: List[int], 
                             current_idx: int) -> Tuple[bool, int]:
        """Check if checkpoint should be saved based on tokens seen."""
        if current_idx < len(save_points) and tokens_seen >= save_points[current_idx]:
            return True, current_idx + 1
        return False, current_idx

