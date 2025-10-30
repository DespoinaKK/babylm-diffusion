import argparse
import json
from pathlib import Path
from datetime import datetime


class ArgumentParser:
    """Centralized argument parsing with JSON config support."""

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="BERT-style model training with various masking strategies"
        )

        parser.add_argument(
            "--config_file", type=Path, help="JSON file for general training parameters"
        )

        # Core arguments
        parser.add_argument("--name", default="test", type=str, help="Name of the run")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        parser.add_argument("--max_steps", type=int, default=0, help="Total training steps, let 0 for auto-calculation")
        parser.add_argument("--more_steps", type=int, default=0, help="Additional training steps")
        parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=True)

        # Model & data arguments
        parser.add_argument("--train_path", type=Path, help="Training data path")
        parser.add_argument("--valid_path", type=Path, help="Validation data path")
        parser.add_argument("--config_file_bert", type=Path, help="JSON config file")
        parser.add_argument("--tokenizer_path", type=Path, help="Tokenizer path")
        parser.add_argument("--output_dir", type=Path, help="Output checkpoint directory")
        parser.add_argument("--seq_length", type=int, help="Sequence length for training")
        parser.add_argument('--n_special_tokens', default=16, type=int, help="Number of special tokens.")
        parser.add_argument(
            "--architecture",
            type=str,
            default="elc-nodwa",
            choices=[
                "ltg", "acn", "hybrid", "clamped_hybrid", "elc", "sw",
                "sparse", "dynamic", "gmlp_transformer", "simple_rope",
                "mswa", "elc-nodwa"
            ],
            help="Model architecture choice"
        )

        # Optimization
        parser.add_argument("--optimizer", type=str, default="lamb")
        parser.add_argument("--optimizer_eps", default=1e-8, type=float, help="Optimizer epsilon.")
        parser.add_argument("--optimizer_beta1", type=float, default=0.9)
        parser.add_argument("--optimizer_beta2", type=float, default=0.98)
        parser.add_argument("--learning_rate", type=float, default=0.007)
        parser.add_argument("--weight_decay", type=float, default=0.1)
        parser.add_argument("--local_batch_size", type=int, default=32)
        parser.add_argument("--global_batch_size", type=int, default=128)
        parser.add_argument("--warmup_proportion", type=float, default=0.016)
        parser.add_argument("--cooldown_proportion", type=float, default=0.016)
        parser.add_argument("--ema_decay", type=float, default=0.999)
        parser.add_argument("--max_gradient", type=float, default=2.0)

        # Masking
        parser.add_argument("--mask_random_p", type=float, default=0.0)
        parser.add_argument("--mask_keep_p", type=float, default=0.0)

        parser.add_argument("--cosine_squared", action="store_true", help="Enable cosine squared schedule")
        parser.add_argument("--cosine_linear", action="store_true", help="Enable cosine linear schedule")
        parser.add_argument("--uniform_schedule", action="store_true", help="Enable linear schedule")
        parser.add_argument("--gaussian_schedule", action="store_true", help="Enable Gaussian schedule")

        parser.add_argument(
            "--bimodal_gaussian_schedule",
            action="store_true",
            help="Enable bimodal Gaussian schedule"
        )

        parser.add_argument("--bimodal_left_mean", type=float, default=0.15)
        parser.add_argument("--bimodal_left_std", type=float, default=0.05)
        parser.add_argument("--bimodal_right_initial", type=float, default=0.3)
        parser.add_argument("--bimodal_right_final", type=float, default=0.9)
        parser.add_argument("--bimodal_right_std", type=float, default=0.1)
        parser.add_argument("--bimodal_exp_rate", type=float, default=2.0)
        parser.add_argument("--bimodal_weight_left", type=float, default=0.4)
        parser.add_argument("--gaussian_power", type=float, default=0.0)

        # Experimental
        parser.add_argument("--use_frequency_masking", action="store_true")
        parser.add_argument("--frequency_method", type=str, default="rank", choices=["rank", "inverse", "exponential"])
        parser.add_argument("--token_freq_distribution_path", type=Path)
        parser.add_argument("--load_cached_frequencies", action="store_true")
        parser.add_argument("--pow_frequency", type=float, default=0.02)
        parser.add_argument("--mask_curriculum", action="store_true")
        parser.add_argument("--time_conditioning", action="store_true")
        parser.add_argument("--cond_dim", type=int)

        # Logging / save
        parser.add_argument("--validate_every", type=int, default=1000)
        parser.add_argument("--save_every", type=int, default=1000)
        parser.add_argument("--checkpoint_filename", type=str, default=None)

        return parser

    @staticmethod
    def parse_and_setup() -> argparse.Namespace:
        """Parse CLI args, merge with JSON config, and setup directories."""
        parser = ArgumentParser.create_parser()

        # Parse known args first to get config_file
        args, unknown = parser.parse_known_args()

        # Load JSON config
        if args.config_file and args.config_file.exists():
            with open(args.config_file, "r") as f:
                config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(args, key):
                    # Convert to bool if the CLI arg is a store_true flag
                    cli_type = type(getattr(args, key))
                    if cli_type is bool:
                        setattr(args, key, bool(value))
                    else:
                        setattr(args, key, value)

        # Check required args
        if args.seq_length is None:
            raise ValueError("--seq_length must be provided either in JSON or CLI")

        # Setup output directory
        args.output_dir = Path(args.output_dir)
        args.output_path = args.output_dir / args.name / datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path.mkdir(parents=True, exist_ok=True)

        return args

