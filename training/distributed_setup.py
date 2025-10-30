import torch
import torch.distributed as dist
from socket import gethostname
from tokenizers import Tokenizer
import argparse
import os

from data.dataset_utils import DatasetAnalyzer, TrainingStepsCalculator
from utils.utils import seed_everything, is_main_process, get_rank, get_world_size

class DistributedTrainingSetup:
    """Handles distributed training setup and configuration."""
    
    @staticmethod
    def setup(args: argparse.Namespace, tokenizer: Tokenizer) -> None:
        """Setup distributed training environment."""
        assert torch.cuda.is_available(), "CUDA is required"
        
        # Get distributed training parameters
        args.n_gpu = torch.cuda.device_count()
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        
        assert args.gpus_per_node == torch.cuda.device_count()
        
        print(f"Hello from rank {args.rank} of {args.world_size} on {gethostname()} "
              f"with {args.gpus_per_node} GPUs", flush=True)
        
        # Set dataset type and seed
        seed_everything(args.seed + args.rank)
        
        # Initialize process group
        torch.distributed.init_process_group(backend="nccl", rank=args.rank, world_size=args.world_size)
        
        if args.rank == 0:
            print(f"Process group initialized: {torch.distributed.is_initialized()}", flush=True)
        
        # Setup CUDA device
        args.local_rank = args.rank - args.gpus_per_node * (args.rank // args.gpus_per_node)
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        
        print(f"Device setup - host: {gethostname()}, rank: {args.rank}, "
              f"local_rank: {args.local_rank}")
        
        # Calculate training parameters
        total_tokens = DatasetAnalyzer.count_tokens(args.train_path)
        print(f"Total tokens: {total_tokens:,}")
        
        args.max_steps = TrainingStepsCalculator.calculate_for_epochs(args, tokenizer) + args.more_steps
        args.vocab_size = tokenizer.get_vocab_size()
        
        if is_main_process():
            DistributedTrainingSetup._print_training_summary(args)
    
    @staticmethod
    def _print_training_summary(args: argparse.Namespace) -> None:
        """Print training configuration summary."""
        total_instances = (args.max_steps * get_world_size() * 
                          args.local_batch_size * args.seq_length)
        
        print(f"\nTraining Summary:")
        print(f"  Steps: {args.max_steps:,}")
        print(f"  GPUs: {get_world_size()}")
        print(f"  Batch size per GPU: {args.local_batch_size:,}")
        print(f"  Sequence length: {args.seq_length:,}")
        print(f"  Total subword instances: {total_instances:,}")
        print(f"  Vocabulary size: {args.vocab_size:,}")
