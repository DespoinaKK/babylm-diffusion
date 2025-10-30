import torch
import csv
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
import argparse
from tokenizers import Tokenizer
import os

from tqdm import tqdm

from utils.utils import is_main_process, get_world_size
from data.dataset import GenericDataset

class FrequencyMaskingSetup:
    """Handles frequency-based masking setup and distribution computation."""
    
    @staticmethod
    def setup_distributed(args: argparse.Namespace, tokenizer: Tokenizer, 
                         epoch: int = 0) -> Optional[torch.Tensor]:
        """Setup frequency masking for distributed training."""
        if not getattr(args, 'use_frequency_masking', False):
            print("Frequency masking disabled")
            return None
        
        freq_path = getattr(args, 'token_freq_distribution_path', 'token_freq_distribution.pt')
        
        # Try loading from cache
        if os.path.exists(freq_path) and getattr(args, 'load_cached_frequencies', True):
            if is_main_process():
                print(f"Loading cached frequency distribution from {freq_path}")
            
            freq_data = torch.load(freq_path, map_location='cpu')
            token_freq_distribution = freq_data['token_freq_distribution']
            
            if is_main_process():
                print(f"Loaded frequency distribution: {token_freq_distribution.shape}")
            
            return token_freq_distribution
        
        # Compute on main process
        if is_main_process():
            print("Computing frequency distribution from scratch...")
            token_freq_distribution = FrequencyMaskingSetup.compute_distribution(
                args, tokenizer,
                method=getattr(args, 'frequency_method', 'inverse'),
                frequency_scale=getattr(args, 'frequency_scale', 1.0),
                normalize=True
            )
            
            # Save the distribution
            torch.save({
                'token_freq_distribution': token_freq_distribution,
                'method': getattr(args, 'frequency_method', 'inverse'),
                'frequency_scale': getattr(args, 'frequency_scale', 1.0),
                'vocab_size': args.vocab_size,
                'total_processes': get_world_size(),
            }, freq_path)
            
            print(f"Frequency distribution saved to {freq_path}")
        else:
            token_freq_distribution = None
        
        # Synchronize processes
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            
            if not is_main_process():
                if os.path.exists(freq_path):
                    freq_data = torch.load(freq_path, map_location='cpu')
                    token_freq_distribution = freq_data['token_freq_distribution']
                else:
                    raise FileNotFoundError(f"Frequency distribution file not found: {freq_path}")
        
        if is_main_process():
            print("Frequency distribution setup complete for all processes!")
        
        return token_freq_distribution
    
    @staticmethod
    def compute_distribution(args: argparse.Namespace, tokenizer: Tokenizer, 
                           method: str = 'inverse', frequency_scale: float = 1.0, 
                           normalize: bool = True) -> Optional[torch.Tensor]:
        """Compute token frequency distribution for masking weights."""
        if method == 'uniform':
            return torch.ones(args.vocab_size, dtype=torch.float32)
        
        if not is_main_process():
            return None
        
        print("=" * 80)
        print("COMPUTING TOKEN FREQUENCY DISTRIBUTION")
        print("=" * 80)
        
        # Create dataset for frequency computation
        dataset = GenericDataset(args.train_path, tokenizer, args, args.seq_length, 
                               rank=0, world_size=1)
        
        dataloader = DataLoader(
            dataset, shuffle=False, batch_size=args.local_batch_size,
            num_workers=0, drop_last=False, pin_memory=False
        )
        
        print(f"Counting tokens across {len(dataloader)} batches...")
        
        # Count frequencies
        token_counts = torch.zeros(args.vocab_size, dtype=torch.long)
        total_tokens = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing frequencies")):
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"Processed {batch_idx}/{len(dataloader)} batches, "
                      f"total tokens: {total_tokens:,}")
            
            input_ids = FrequencyMaskingSetup._extract_input_ids(batch)
            unique_tokens, counts = torch.unique(input_ids.flatten(), return_counts=True)
            
            for token, count in zip(unique_tokens, counts):
                if 0 <= token < args.vocab_size:
                    token_counts[token] += count
                    total_tokens += count
        
        print(f"Total tokens processed: {total_tokens:,}")
        
        # Convert to weights based on method
        freq_weights = FrequencyMaskingSetup._compute_weights(
            token_counts, total_tokens, method, frequency_scale, args.vocab_size
        )
        
        if normalize and method != 'uniform':
            freq_weights = freq_weights / freq_weights.mean()
        
        FrequencyMaskingSetup._print_statistics(freq_weights, token_counts)
        
        return freq_weights
    
    @staticmethod
    def _extract_input_ids(batch) -> torch.Tensor:
        """Extract input_ids from different batch formats."""
        if isinstance(batch, dict):
            return batch['input_ids']
        elif isinstance(batch, (list, tuple)):
            return batch[0]
        else:
            return batch
    
    @staticmethod
    def _compute_weights(token_counts: torch.Tensor, total_tokens: int, 
                        method: str, frequency_scale: float, vocab_size: int) -> torch.Tensor:
        """Compute frequency weights based on the specified method."""
        token_frequencies = token_counts.float() / total_tokens
        
        if method == 'inverse':
            epsilon = 1e-8
            return 1.0 / (token_frequencies + epsilon)
        
        elif method == 'rank':
            _, sorted_indices = torch.sort(token_frequencies, descending=True)
            token_ranks = torch.zeros_like(sorted_indices, dtype=torch.float32)
            token_ranks[sorted_indices] = torch.arange(len(sorted_indices), dtype=torch.float32)
            return (token_ranks + 1) / vocab_size
        
        elif method == 'exponential':
            epsilon = 1e-8
            return torch.exp(-torch.log(token_frequencies + epsilon) * frequency_scale)
        
        else:
            raise ValueError(f"Unknown frequency method: {method}")
    
    @staticmethod
    def _print_statistics(freq_weights: torch.Tensor, token_counts: torch.Tensor) -> None:
        """Print frequency distribution statistics."""
        print(f"Distribution stats - Mean: {freq_weights.mean():.4f}, "
              f"Std: {freq_weights.std():.4f}, "
              f"Min: {freq_weights.min():.4f}, Max: {freq_weights.max():.4f}")
        
        token_frequencies = token_counts.float() / token_counts.sum()
        top_frequent = torch.argsort(token_frequencies, descending=True)[:5]
        top_rare = torch.argsort(token_frequencies, descending=False)[:5]
        
        print("\nTop 5 frequent tokens and weights:")
        for idx in top_frequent:
            if token_counts[idx] > 0:
                print(f"  Token {idx}: freq={token_frequencies[idx]:.6f}, "
                      f"weight={freq_weights[idx]:.4f}")
        
        print("\nTop 5 rare tokens and weights:")
        for idx in top_rare:
            if token_counts[idx] > 0:
                print(f"  Token {idx}: freq={token_frequencies[idx]:.6f}, "
                      f"weight={freq_weights[idx]:.4f}")


