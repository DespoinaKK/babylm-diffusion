import math
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, Optional, Any
from tokenizers import Tokenizer
import argparse

from data.dataset import GenericDataset, ValidationDataset
from masking.frequency_masking import FrequencyMaskingSetup
from utils.utils import is_main_process, get_rank, get_world_size

class DatasetManager:
    """Manages dataset loading and configuration."""
    
    @staticmethod
    def load_datasets(args: argparse.Namespace, tokenizer: Tokenizer, epoch: int, 
                     global_step: int, train_dataloader, valid_dataloader,
                     token_freq_distribution: Optional[torch.Tensor] = None) -> Tuple[DataLoader, DataLoader, Optional[torch.Tensor]]:
        """Load datasets with appropriate configuration for current training phase."""
                
        # Setup frequency masking if needed
        if token_freq_distribution is None and getattr(args, 'use_frequency_masking', False):
            if is_main_process():
                print("\nSetting up frequency masking...")
            token_freq_distribution = FrequencyMaskingSetup.setup_distributed(args, tokenizer, epoch)
            args.token_freq_distribution = token_freq_distribution
        
        # Determine sequence length and batch size based on training progress
        seq_length, global_batch_size = DatasetManager._get_training_config(args, global_step)
        
        # Load training dataset if needed
        if (train_dataloader is None or 
            train_dataloader.dataset.seq_length != seq_length):
            
            train_data = GenericDataset(
                args.train_path, tokenizer, args, seq_length, args.rank, args.world_size
            )
            if is_main_process():
                train_data.show_random_item(tokenizer)
        else:
            train_data = train_dataloader.dataset
        
        # Setup batch configuration
        args.current_global_batch_size = global_batch_size
        total_local_batch_size = int(args.current_global_batch_size / args.world_size + 0.5)
        args.accumulate_steps = int(math.ceil(total_local_batch_size / args.local_batch_size))
        args.current_local_batch_size = total_local_batch_size // args.accumulate_steps
        
        # Create training dataloader
        train_seed = args.seed + get_rank() + epoch * get_world_size()
        train_dataloader = DataLoader(
            train_data,
            shuffle=True,
            batch_size=args.current_local_batch_size,
            num_workers=0,
            generator=torch.Generator().manual_seed(train_seed),
            drop_last=True,
            pin_memory=True,
        )
        
        # Create validation dataloader if needed
        if valid_dataloader is None:
            valid_data = ValidationDataset(args.valid_path, tokenizer, args)
            valid_data = torch.utils.data.Subset(valid_data, range(len(valid_data)))
            valid_dataloader = DataLoader(
                valid_data,
                shuffle=False,
                batch_size=16,
                num_workers=0,
                generator=torch.Generator().manual_seed(42),
                drop_last=True,
                pin_memory=True,
            )
                
        return train_dataloader, valid_dataloader, token_freq_distribution
    
    @staticmethod
    def _get_training_config(args: argparse.Namespace, global_step: int) -> Tuple[int, int]:
        """Get sequence length and batch size based on training progress."""
        return args.seq_length, args.global_batch_size
    