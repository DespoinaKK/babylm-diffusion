import os
from itertools import count
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from config.arguments import ArgumentParser

from data.dataset_manager import DatasetManager
from data.dataset import ValidationDataset


from training.distributed_setup import DistributedTrainingSetup
from training.model_setup import ModelSetup
from training.training_loop import TrainingLoop
from training.checkpoint_manager import CheckpointManager
from training.validation import ValidationManager

from utils.utils import is_main_process

from model import Bert

def main():
    """Main training function."""
    # Parse arguments and setup configuration
    args = ArgumentParser.parse_and_setup()
            
    # Setup tokenizer and distributed training environment
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    DistributedTrainingSetup.setup(args, tokenizer)
    
    # Prepare model, optimizer, scheduler, and EMA
    model, ema, optimizer, scheduler, global_step, start_epoch = ModelSetup.prepare_model_and_optimizer(args)
    
    # Initialize dataloaders (will be populated during training)
    train_dataloader, valid_dataloader = None, None
    torch.cuda.empty_cache()
    
    # Setup development dataloaders for evaluation
    dev_paths = [
        '../babyLM_2025_data/dev/bnc_spoken_100M-correct_tokenized.bin',
        '../babyLM_2025_data/dev/childes_100M-correct_tokenized.bin',
        '../babyLM_2025_data/dev/gutenberg_100M-correct_tokenized.bin',
        '../babyLM_2025_data/dev/open_subtitles_100M-correct_tokenized.bin',
        '../babyLM_2025_data/dev/simple_wiki_100M-correct_tokenized.bin',
        '../babyLM_2025_data/dev/switchboard_100M-correct_tokenized.bin'
    ]
    
    dev_dataloaders = []
    for dev_path in dev_paths:
        try:
            dev_data = ValidationDataset(dev_path, tokenizer, args)
            dev_dataloader = DataLoader(
                dev_data, 
                shuffle=False, 
                batch_size=16, 
                generator=torch.Generator().manual_seed(42),
                drop_last=True, 
                pin_memory=True
            )
            dev_dataloaders.append(dev_dataloader)
        except Exception as e:
            print(f"Warning: Could not load dev dataset {dev_path}: {e}")
    
    # Initialize frequency masking distribution (if used)
    token_freq_distribution = None
    
    # Main training loop over epochs
    for epoch in count(start=start_epoch):
        # Load datasets for current epoch
        train_dataloader, valid_dataloader, token_freq_distribution = DatasetManager.load_datasets(
            args, 
            tokenizer, 
            epoch, 
            global_step, 
            train_dataloader, 
            valid_dataloader, 
            token_freq_distribution
        )
        
        # Run training epoch
        global_step = TrainingLoop.training_epoch_vector(
            model, 
            ema, 
            train_dataloader, 
            valid_dataloader, 
            optimizer, 
            scheduler,
            global_step, 
            epoch, 
            args, 
            tokenizer, 
            dev_dataloaders=dev_dataloaders, 
        )
        
        # Check if training is complete
        if global_step >= args.max_steps:
            break
    
    # Save final checkpoint
    CheckpointManager.save_checkpoint(model, ema, optimizer, scheduler, global_step, args)
    
    if is_main_process():
        print("Training completed successfully!")


if __name__ == "__main__":
    main()