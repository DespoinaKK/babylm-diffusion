import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from pathlib import Path
from typing import Tuple, Optional, Any
import json
import argparse

from optimization.lamb import Lamb
from training.ema import ExponentialMovingAverage
from utils.utils import cosine_schedule_with_warmup_cooldown, apply_mask_to_weights, is_main_process

from model import Bert

class ConfigLoader:
    """Loads and applies model configuration."""
    
    @staticmethod
    def load_and_apply(args: argparse.Namespace) -> argparse.Namespace:
        """Load config from JSON file and apply to args."""
        config_file_bert = Path(args.config_file_bert)

        try:
            with config_file_bert.open("r") as f:
                config = json.load(f)
            
            for key, value in config.items():
                setattr(args, key, value)
            
            print(f"Loaded config from {args.config_file_bert}")
            return args
            
        except Exception as e:
            print(f"Error loading config: {e}")
            raise


class ModelSetup:
    """Handles model creation and optimization setup."""
    
    @staticmethod
    def prepare_model_and_optimizer(args: argparse.Namespace) -> Tuple[nn.Module, ExponentialMovingAverage, 
                                                                     torch.optim.Optimizer, Any, int, int]:
        """Prepare model, EMA, optimizer, scheduler and load checkpoints if available."""        
        args_bert = ConfigLoader.load_and_apply(args)
        print(f"Model configuration: {args_bert}", flush=True)
        model = Bert(args_bert)
        
        if is_main_process():
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(model)
            print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)
        
        model.to(args.device)
        
        # Setup optimizer
        optimizer = ModelSetup._create_optimizer(model, args)
        
        # Setup scheduler
        scheduler = cosine_schedule_with_warmup_cooldown(
            optimizer,
            int(args.max_steps * args.warmup_proportion),
            int(args.max_steps * args.cooldown_proportion),
            args.max_steps,
            0.1
        )
        
        # Setup DDP
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            bucket_cap_mb=torch.cuda.get_device_properties(args.device).total_memory,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True
        )
        
        # Setup EMA
        ema = ExponentialMovingAverage(
            parameters=model.parameters(),
            decay=args.ema_decay,
            use_num_updates=True
        )
        ema.move_shadow_params_to_device(args.device)
        
        # Load checkpoint if specified
        global_step, epoch = ModelSetup._load_checkpoint_if_exists(
            args, model, optimizer, scheduler, ema
        )
        
        return model, ema, optimizer, scheduler, global_step, epoch
    
    @staticmethod
    def _create_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
        """Create optimizer with proper parameter grouping."""
        no_decay = ['bias', 'layer_norm']
        decay_params = [(n, p) for n, p in model.named_parameters() 
                       if not any(nd in n for nd in no_decay)]
        no_decay_params = [(n, p) for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)]
        
        optimizer_grouped_parameters = [
            {'params': [p for _, p in decay_params], 'weight_decay': args.weight_decay},
            {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0}
        ]
        
        if is_main_process():
            print("Parameters without weight decay:")
            for n, _ in no_decay_params:
                print(n)
            print("\nParameters with weight decay:")
            for n, _ in decay_params:
                print(n)
            print(flush=True)
        
        if args.optimizer.lower() in ["adam", "adamw"]:
            return torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1, args.optimizer_beta2),
                eps=args.optimizer_eps,
            )
        elif args.optimizer.lower() == "lamb":
            return Lamb(
                optimizer_grouped_parameters,
                args.learning_rate,
                betas=(args.optimizer_beta1, args.optimizer_beta2),
                eps=args.optimizer_eps,
            )
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    @staticmethod
    def _load_checkpoint_if_exists(args: argparse.Namespace, model: nn.Module, 
                                 optimizer: torch.optim.Optimizer, scheduler: Any,
                                 ema: ExponentialMovingAverage) -> Tuple[int, int]:
        """Load checkpoint if specified and return global_step, epoch."""
        global_step, epoch = 0, 0
        
        if args.checkpoint_filename is not None:
            if is_main_process():
                print(f"Loading checkpoint from {args.checkpoint_filename}")
            
            state_dict = torch.load(args.checkpoint_filename, map_location="cpu")
            model.load_state_dict(state_dict["model"])
            optimizer.load_state_dict(state_dict["optimizer"])
            scheduler.load_state_dict(state_dict["scheduler"])
            global_step = state_dict["global_step"]
            epoch = state_dict["epoch"]
            
            # Load EMA state
            if "ema" in state_dict:
                ema.load_state_dict(state_dict["ema"])
            elif "ema_model" in state_dict:
                print("Converting old ema_model checkpoint to new EMA format...")
                old_ema_params = [p for p in state_dict["ema_model"].values() 
                                if isinstance(p, torch.Tensor)]
                if len(old_ema_params) == len(ema.shadow_params):
                    for shadow_param, old_param in zip(ema.shadow_params, old_ema_params):
                        shadow_param.data.copy_(old_param.data)
                    print("Successfully converted old EMA checkpoint!")
                else:
                    print("Warning: Could not convert old EMA checkpoint")
            else:
                print("Warning: No EMA state found in checkpoint")
            
            if is_main_process():
                print(f"Resumed from step {global_step}, epoch {epoch}")
        
        return global_step, epoch
