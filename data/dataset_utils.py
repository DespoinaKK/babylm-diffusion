import math
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from utils.utils import is_main_process
from data.dataset import GenericDataset

class DatasetAnalyzer:
    """Utilities for analyzing dataset properties."""
    
    @staticmethod
    def count_tokens(tokenized_file_path: Path) -> int:
        """Count total tokens in a tokenized dataset file."""
        try:
            tokenized_documents = torch.load(tokenized_file_path)
            return sum(len(doc) for doc in tokenized_documents)
        except Exception as e:
            print(f"Error counting tokens in {tokenized_file_path}: {e}")
            return 0



class TrainingStepsCalculator:
    """Calculates training steps for multi-epoch training."""
    
    @staticmethod
    def calculate_for_epochs(args: argparse.Namespace, tokenizer: Tokenizer, 
                           num_epochs: int = 10) -> int:
        """Calculate steps for specified number of epochs."""
        if hasattr(args, 'enable_multi_length') and args.enable_multi_length:
            raise NotImplementedError("Multi-length training step calculation not implemented")
        
        return TrainingStepsCalculator._calculate_standard(args, tokenizer, num_epochs)
    
    @staticmethod
    def _calculate_standard(args: argparse.Namespace, tokenizer: Tokenizer, 
                          num_epochs: int) -> int:
        """Standard calculation for fixed sequence length training."""
        rank, world_size = args.rank, args.world_size
        
        # Define phases with different sequence lengths and batch sizes
        phases = [(args.seq_length, args.global_batch_size, num_epochs)]
        
        total_steps = 0
        if is_main_process():
            print("Calculating steps for each phase:")
        
        for i, (seq_len, global_batch, epochs) in enumerate(phases):
            temp_data = GenericDataset(args.train_path, tokenizer, args, seq_len, rank, world_size)
            
            total_local_batch = int(global_batch / world_size + 0.5)
            accumulate_steps = int(math.ceil(total_local_batch / args.local_batch_size))
            current_local_batch = total_local_batch // accumulate_steps
            
            temp_dataloader = DataLoader(
                temp_data, shuffle=True, batch_size=current_local_batch,
                num_workers=0, generator=torch.Generator().manual_seed(42),
                drop_last=True, pin_memory=True
            )
            
            steps_per_epoch = len(temp_dataloader) // accumulate_steps
            phase_steps = steps_per_epoch * epochs
            total_steps += phase_steps
            
            if is_main_process():
                print(f" Phase {i+1}: seq_len={seq_len}, global_batch={global_batch}, "
                      f"steps_per_epoch={steps_per_epoch}, epochs={epochs}, "
                      f"phase_steps={phase_steps}")
            
            del temp_dataloader, temp_data
            torch.cuda.empty_cache()
        
        return total_steps