import torch
from typing import Tuple, Any, Optional
from tokenizers import Tokenizer
import argparse

from masking.masking_processor import MaskingProcessor


class BatchProcessor:
    """Handles batch processing and data loading."""
    
    @staticmethod
    def get_batch(dataloader, device: torch.device, global_step: int, train: bool = True,
                 tokenizer: Optional[Tokenizer] = None, args: Optional[argparse.Namespace] = None,
                 mask_p: Optional[torch.Tensor] = None, epoch: int = 0) -> Tuple[torch.Tensor, ...]:
        """Fetch a batch, apply masking, and move tensors to device."""
        batch = next(dataloader)
        
        if len(batch) == 2:  # Format: (input_ids, attention_mask)
            input_ids, attention_mask = [t.pin_memory().to(device, non_blocking=True) for t in batch]
            input_ids = input_ids.t()
            
            if train and tokenizer and args and mask_p is not None:
                if args.use_frequency_masking and hasattr(args, 'token_freq_distribution'):
                    input_ids, target_ids, mask_p = MaskingProcessor.apply_frequency_based_masking(
                        input_ids.t(), tokenizer, args, global_step, mask_p, epoch, 
                        args.token_freq_distribution
                    )
                else:
                    input_ids, target_ids, mask_p = MaskingProcessor.apply_random_masking_vector(
                        input_ids.t(), tokenizer, args, global_step, mask_p
                    )
                return input_ids.t(), attention_mask, target_ids.t(), mask_p
            else:
                raise ValueError("Invalid masking configuration in get_batch")
        else:  # Original dataset format (Validation dataset)
            input_ids, target_ids, attention_mask, mask_p = [
                t.pin_memory().to(device, non_blocking=True) for t in batch
            ]
            return input_ids.t(), attention_mask, target_ids.t(), mask_p

