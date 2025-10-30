import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List, Any
from tokenizers import Tokenizer

from masking.noise_schedules import MaskingStrategies
from masking.batch_processing import BatchProcessor

from training.checkpoint_manager import CheckpointManager
from training.validation import ValidationManager
from training.ema import ExponentialMovingAverage

from utils.utils import is_main_process


class UtilityFunctions:
    """Utility functions for training."""
    
    @staticmethod
    def all_reduce_scalar(x: float) -> float:
        """Sum a scalar across all processes and return the result."""
        t = torch.tensor(x, device='cuda')
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return t.item()
    
    @staticmethod
    def compute_total_valid_tokens(train_dataloader: DataLoader, tokenizer: Tokenizer) -> int:
        """Compute total valid (non-pad) tokens across all GPUs."""
        pad_id = torch.tensor(tokenizer.token_to_id("<pad>"))
        total_local = sum((seq != pad_id).sum().item() for seq, _ in train_dataloader._dataset)
        return UtilityFunctions.all_reduce_scalar(total_local)



class TrainingLoop:
    """Main training loop implementation."""
    
    @staticmethod
    def training_epoch_vector(model: nn.Module, ema: ExponentialMovingAverage,
                             train_dataloader: DataLoader, valid_dataloader: DataLoader,
                             optimizer: torch.optim.Optimizer, scheduler: Any,
                             global_step: int, epoch: int, args: argparse.Namespace,
                             tokenizer: Tokenizer, dev_dataloaders: Optional[List[DataLoader]] = None,
                             csv_output_path: Optional[str] = None) -> int:
        """Perform one training epoch with gradient accumulation, EMA updates, and checkpoint saving."""
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        # Setup
        num_steps = min(len(train_dataloader), (args.max_steps - global_step) * args.accumulate_steps)
        train_dataloader = iter(train_dataloader)
        batch_size = args.local_batch_size
        
        # Metrics tracking
        total_loss = total_accuracy = total_mask_p = total_grad_norm = 0.0
        
        # Checkpoint setup for epoch 0
        tokens_seen_global = current_save_idx = 0
        save_points = save_percentages = []
        total_tokens = None
        if epoch == 0:
            total_tokens = UtilityFunctions.compute_total_valid_tokens(train_dataloader, tokenizer)
            save_points, save_percentages = CheckpointManager.get_save_points(total_tokens)
            if is_main_process():
                print(f"[Epoch 0] Total valid tokens: {total_tokens:,}")
                print(f"[Epoch 0] Save points: {save_points}")
        
        # Prepare first batch
        batch_mask_p, loss_weights = MaskingStrategies.get_batch_mask_probabilities_and_weights(
            args, global_step, args.device, batch_size
        )
        input_ids, attention_mask, target_ids, mask_p = BatchProcessor.get_batch(
            train_dataloader, args.device, global_step,
            tokenizer=tokenizer, args=args, mask_p=batch_mask_p, epoch=epoch
        )
        
        progress_bar = tqdm(total=args.max_steps, desc="Train", initial=global_step, 
                           disable=not is_main_process())
        
        # Main training loop
        for local_step in range(num_steps):
            with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
                loss, accuracy, num_tokens = model(
                    input_ids, attention_mask, target_ids, mask_p=mask_p, sum=True
                )
            
            # Loss weighting
            weight = 1.0 / args.accumulate_steps
            
            # Backpropagation
            sequence_scaled_loss = (loss * loss_weights).sum()
            
            batch_tokens = max(1, (input_ids != tokenizer.token_to_id("<pad>")).sum().item())
            
            (sequence_scaled_loss * weight / batch_tokens).backward()

            # Update metrics
            
            total_loss += (loss_weights * loss.detach()).sum() * weight / batch_tokens

            total_accuracy += accuracy * weight
            total_mask_p += mask_p.mean() * weight
            
            # Gradient accumulation step
            if (local_step + 1) % args.accumulate_steps == 0:
                # Gradient clipping and optimization
                total_grad_norm += nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient) * weight
                optimizer.step()
                scheduler.step()
                ema.update(model.parameters())
                optimizer.zero_grad(set_to_none=True)
                
                # Progress tracking
                progress_bar.update(1)
                progress_bar.set_postfix(global_step=global_step)
                
                # Checkpoint saving for epoch 0
                if epoch == 0:
                    tokens_seen_global += UtilityFunctions.all_reduce_scalar(batch_tokens)
                    should_save, current_save_idx = CheckpointManager.should_save_checkpoint(
                        tokens_seen_global, save_points, current_save_idx
                    )
                    if should_save:
                        percentage = save_percentages[current_save_idx - 1]
                        print(f"🔥 Saving checkpoint at {percentage}% tokens "
                              f"({tokens_seen_global}/{total_tokens}), step {global_step}")
                        # Uncomment to enable checkpoint saving:
                        # CheckpointManager.save_checkpoint(model, ema, optimizer, scheduler, 
                        #                                 global_step, epoch, args)
                
                # Regular checkpoint saving
                if global_step % args.save_every == 0:
                    CheckpointManager.save_checkpoint(model, ema, optimizer, scheduler, 
                                                    global_step, args)
                
                # EMA validation
                if (global_step + 1) % args.validate_every == 0:
                    ValidationManager.run_ema_validation(
                        model, ema, valid_dataloader, epoch, args,
                        dev_dataloaders=dev_dataloaders, csv_output_path=csv_output_path
                    )
                
                # Print metrics
                if is_main_process():
                    print(f"Epoch {epoch} | Step {global_step} | "
                          f"Loss: {total_loss:.4f} | "
                          f"Acc: {total_accuracy * 100:.2f}% | GradNorm: {total_grad_norm:.4f}")
                
                # Reset metrics and increment step
                total_loss = total_accuracy = total_mask_p = total_grad_norm = 0.0
                global_step += 1
            
            # Prepare next batch (except last step)
            if local_step < num_steps - 1:
                batch_mask_p, loss_weights = MaskingStrategies.get_batch_mask_probabilities_and_weights(
                    args, global_step, args.device, batch_size
                )
                input_ids, attention_mask, target_ids, mask_p = BatchProcessor.get_batch(
                    train_dataloader, args.device, global_step,
                    tokenizer=tokenizer, args=args, mask_p=batch_mask_p, epoch=epoch
                )
            
            if global_step >= args.max_steps:
                return global_step
        
        return global_step
