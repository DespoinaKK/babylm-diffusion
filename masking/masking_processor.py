
import torch
from typing import Tuple
from tokenizers import Tokenizer
import argparse

class MaskingProcessor:
    """Handles token masking operations."""
    
    @staticmethod
    def apply_random_masking_vector(input_ids: torch.Tensor, tokenizer: Tokenizer,
                                  args: argparse.Namespace, global_step: int,
                                  mask_p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random masking with per-sequence probabilities."""
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        mask_token_id = tokenizer.token_to_id("<mask>")
        
        # Random values per token; never mask special tokens
        token_random_values = torch.rand(batch_size, seq_len, device=device)
        token_random_values[input_ids < args.n_special_tokens] = 1.0
        
        # Decide which tokens to mask
        should_mask = token_random_values < mask_p.unsqueeze(1)

        actual_mask_p = should_mask.float().mean(dim=1)

        
        # Replacement selection
        # Set mask_random_p, or mask_keep_p to nonzero values to activate
        # Default behavior for regular diffusion model: does nothing
        replacement_p = torch.rand(batch_size, seq_len, device=device)
        random_mask = should_mask & (replacement_p < args.mask_random_p)
        keep_mask = should_mask & (replacement_p > 
            args.mask_random_p) & (replacement_p < args.mask_random_p + args.mask_keep_p)
        mask_tokens = should_mask & ~(random_mask | keep_mask)
        
        # Apply masks
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_tokens] = mask_token_id
        if random_mask.any():
            masked_input_ids[random_mask] = torch.randint(
                args.n_special_tokens, args.vocab_size, 
                (random_mask.sum().item(),), device=device
            )
        
        # Target IDs
        target_ids = torch.where(should_mask, input_ids, -100)
        
        return masked_input_ids, target_ids, actual_mask_p
    
    @staticmethod
    def apply_frequency_based_masking(input_ids: torch.Tensor, tokenizer: Tokenizer,
                                    args: argparse.Namespace, global_step: int,
                                    mask_p: torch.Tensor, epoch: int,
                                    token_freq_distribution: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply frequency-aware masking strategy."""
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        mask_token_id = tokenizer.token_to_id("<mask>")
        
        # Compute per-token mask probabilities
        per_token_mask_p = MaskingProcessor._compute_frequency_mask_probabilities(
            input_ids, mask_p, args, epoch, token_freq_distribution.to(device)
        )

        # Apply masking
        token_random_values = torch.rand(batch_size, seq_len, device=device)
        token_random_values[input_ids < args.n_special_tokens] = 1.0
        
        should_mask = token_random_values < per_token_mask_p
        
        replacement_p = torch.rand(batch_size, seq_len, device=device)
        random_mask = should_mask & (replacement_p < args.mask_random_p)
        keep_mask = should_mask & (replacement_p > 
            args.mask_random_p) & (replacement_p < args.mask_random_p + args.mask_keep_p)
        mask_tokens = should_mask & ~(random_mask | keep_mask)
        
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_tokens] = mask_token_id
        if random_mask.any():
            masked_input_ids[random_mask] = torch.randint(
                args.n_special_tokens, args.vocab_size, 
                (random_mask.sum().item(),), device=device
            )
        
        target_ids = torch.where(should_mask, input_ids, -100)
        actual_mask_p = per_token_mask_p.mean(dim=1)
        
        return masked_input_ids, target_ids, actual_mask_p
    
    @staticmethod
    def _compute_frequency_mask_probabilities(input_ids: torch.Tensor, mask_p: torch.Tensor,
                                            args: argparse.Namespace, epoch: int,
                                            token_freq_distribution: torch.Tensor) -> torch.Tensor:
        """Compute per-token mask probabilities based on token frequency."""
        freq_weights = token_freq_distribution[input_ids]
        
        # Normalize to [0,1]
        freq_weights = (freq_weights - freq_weights.min(dim=1, keepdim=True).values) / \
                     (freq_weights.max(dim=1, keepdim=True).values - 
                      freq_weights.min(dim=1, keepdim=True).values + 1e-8)
        
        pow_frequency = (epoch / 9 * args.pow_frequency) if args.mask_curriculum else args.pow_frequency
        freq_weights = torch.pow(freq_weights, pow_frequency)
        
        mask_p_exp = mask_p.unsqueeze(1)
        mean = freq_weights.mean(dim=1).unsqueeze(1)
        
        # Scale based on mean relative to mask_p
        freq_weights = torch.where(
            mean > mask_p_exp,
            freq_weights * (mask_p_exp / (mean + 1e-8)),
            freq_weights
        )
        freq_weights = torch.where(
            mean < mask_p_exp,
            -(1 - freq_weights) * (1 - mask_p_exp) / (1 - mean + 1e-8) + 1,
            freq_weights
        )
        
        return freq_weights