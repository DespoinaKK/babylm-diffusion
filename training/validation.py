import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import csv
from pathlib import Path
from typing import Optional, List, Any, Dict
from tqdm import tqdm
import argparse
import os

from training.ema import ExponentialMovingAverage
from utils.utils import is_main_process
from masking.batch_processing import BatchProcessor

class ValidationManager:
    """Handles model validation and evaluation."""
    
    @staticmethod
    def run_ema_validation(model: nn.Module, ema: ExponentialMovingAverage,
                          valid_dataloader: DataLoader, epoch: int, args: argparse.Namespace,
                          dev_dataloaders: Optional[List[DataLoader]] = None,
                          csv_output_path: Optional[str] = None) -> None:
        """Run validation with EMA weights."""
        if not is_main_process():
            return
        
        print(f"[EMA VALIDATION] Step {epoch}: Starting validation with EMA weights")
        first_param_before = next(model.parameters()).flatten()[:5].clone()
        
        # Switch to EMA weights
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        
        first_param_after = next(model.parameters()).flatten()[:5].clone()
        param_diff = torch.abs(first_param_before - first_param_after).mean()
        # print(f"[EMA VALIDATION] Param diff: {param_diff:.6f}")
        
        # Run validation
        ValidationManager.validation_epoch(
            model, valid_dataloader, epoch, args,
            dev_dataloaders=dev_dataloaders, csv_output_path=csv_output_path
        )
        
        # Restore original weights
        ema.restore(model.parameters())
        first_param_restored = next(model.parameters()).flatten()[:5].clone()
        restore_diff = torch.abs(first_param_before - first_param_restored).mean()
        # print(f"[EMA VALIDATION] Restored diff: {restore_diff:.8f}")
        model.train()
        print("[EMA VALIDATION] Validation complete, resumed training")
    
    @staticmethod
    def validation_epoch(model: nn.Module, valid_dataloader: DataLoader, 
                        epoch: int, args: argparse.Namespace,
                        dev_dataloaders: Optional[List[DataLoader]] = None,
                        csv_output_path: Optional[str] = None) -> None:
        """Run a full validation epoch."""
        model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for step, batch in enumerate(valid_dataloader):                
                input_ids, attention_mask, target_ids, mask_p = BatchProcessor.get_batch(
                    iter([batch]), args.device, 0, train=False
                )
                with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
                    loss, accuracy, _ = model(input_ids, attention_mask, target_ids, mask_p=mask_p, sum=False)
                
                total_loss += loss.mean().item()
                total_accuracy += accuracy
                total_steps += 1
        
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        avg_accuracy = total_accuracy / total_steps if total_steps > 0 else 0.0
        
        print(f"Validation - Epoch {epoch}: Loss {avg_loss:.4f}, Accuracy {avg_accuracy*100:.2f}%")
        
        # Run additional dev set evaluations if provided
        if dev_dataloaders and csv_output_path:
            ValidationManager._evaluate_dev_sets(
                model, dev_dataloaders, epoch, args, csv_output_path
            )
    
    @staticmethod
    def _evaluate_dev_sets(model: nn.Module, dev_dataloaders: List[DataLoader],
                          epoch: int, args: argparse.Namespace, csv_output_path: str) -> None:
        """Evaluate on multiple development sets and save results."""
        dev_names = ['bnc_spoken', 'childes', 'gutenberg', 'open_subtitles', 'simple_wiki', 'switchboard']
        results = []
        
        for i, (dev_dataloader, dev_name) in enumerate(zip(dev_dataloaders, dev_names)):
            total_loss = 0.0
            total_accuracy = 0.0
            total_steps = 0
            
            with torch.no_grad():
                for step, batch in enumerate(dev_dataloader):
                    if step >= 10:  # Limit evaluation steps
                        break
                    
                    input_ids, attention_mask, target_ids, mask_p = BatchProcessor.get_batch(
                        iter([batch]), args.device, 0, train=False
                    )
                    
                    with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
                        loss, accuracy, _ = model(input_ids, attention_mask, target_ids, mask_p=mask_p, sum=False)
                    
                    total_loss += loss.mean().item()
                    total_accuracy += accuracy
                    total_steps += 1
            
            avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
            avg_accuracy = total_accuracy / total_steps if total_steps > 0 else 0.0
            
            results.append({
                'epoch': epoch,
                'dev_set': dev_name,
                'loss': avg_loss,
                'accuracy': avg_accuracy
            })
            
            print(f"Dev {dev_name}: Loss {avg_loss:.4f}, Accuracy {avg_accuracy*100:.2f}%")
        
        # Save results to CSV
        ValidationManager._save_results_to_csv(results, csv_output_path)
    
    @staticmethod
    def _save_results_to_csv(results: List[Dict], csv_path: str) -> None:
        """Save validation results to CSV file."""
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = ['epoch', 'dev_set', 'loss', 'accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for result in results:
                writer.writerow(result)
