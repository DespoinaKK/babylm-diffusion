import torch
import torch.distributions as dist
import math
from typing import Tuple, Optional
import argparse

class MaskingStrategies:
    """Handles different masking probability sampling strategies."""
    
    @staticmethod
    def sample_low_discrepancy(batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample t values using low discrepancy (stratified) sampling."""
        bin_edges = torch.linspace(0.0, 1.0, batch_size + 1, device=device)
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        random_offsets = torch.rand(batch_size, device=device) * bin_widths
        time = bin_edges[:-1] + random_offsets
        return time
    
    @staticmethod
    def get_batch_mask_probabilities_and_weights(args: argparse.Namespace, global_step: int, 
                                               device: torch.device, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate batch mask probabilities and loss weights based on configured strategy."""
        batch_mask_p = torch.zeros(batch_size, device=device)
        loss_weights = torch.zeros(batch_size, device=device)
        
        if args.cosine_squared:
            batch_mask_p, loss_weights = MaskingStrategies._cosine_squared_schedule(
                batch_size, device, global_step
            )
        elif args.cosine_linear:
            batch_mask_p, loss_weights = MaskingStrategies._cosine_linear_schedule(
                batch_size, device, global_step
            )
        elif args.uniform_schedule:
            batch_mask_p, loss_weights = MaskingStrategies._uniform_schedule(
                batch_size, device, global_step
            )
        elif args.gaussian_schedule:
            batch_mask_p, loss_weights = MaskingStrategies._gaussian_schedule(
                args, batch_size, device, global_step
            )
        elif args.bimodal_gaussian_schedule:
            batch_mask_p, loss_weights = MaskingStrategies._bimodal_gaussian_schedule(
                args, batch_size, device, global_step
            )
        else:
            raise ValueError("No valid sampling method specified!")
        
        return batch_mask_p, loss_weights
    
    @staticmethod
    def _cosine_squared_schedule(batch_size: int, device: torch.device, 
                               global_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cosine squared noise schedule."""
        t_values = MaskingStrategies.sample_low_discrepancy(batch_size, device)
        
        # α(t) = cos²(π/2 * (1-t))
        alpha_t = torch.cos(math.pi * 0.5 * (1 - t_values)) ** 2
        batch_mask_p = 1.0 - alpha_t
        
        # Derivatives for loss weighting
        derivatives = math.pi * torch.sin(math.pi * 0.5 * (1 - t_values)) * \
                     torch.cos(math.pi * 0.5 * (1 - t_values))
        loss_weights = torch.abs(derivatives) / (1 - alpha_t + 1e-8)
        
        if global_step % 200 == 0:
            print(f"Cosine squared schedule at step {global_step}, "
                  f"mask_p_range: [{batch_mask_p.min():.4f}, {batch_mask_p.max():.4f}]")
        
        return batch_mask_p, loss_weights
    
    @staticmethod
    def _cosine_linear_schedule(batch_size: int, device: torch.device, 
                              global_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cosine linear noise schedule."""
        t_values = MaskingStrategies.sample_low_discrepancy(batch_size, device)
        
        # α(t) = cos(π/2 * (1-t))
        alpha_t = torch.cos(math.pi * 0.5 * (1 - t_values))
        batch_mask_p = 1.0 - alpha_t
        
        # Derivatives for loss weighting
        derivatives = -math.pi * 0.5 * torch.sin(math.pi * 0.5 * (1 - t_values))
        loss_weights = torch.abs(derivatives) / (1 - alpha_t + 1e-8)
        
        if global_step % 200 == 0:
            print(f"Cosine linear schedule at step {global_step}, "
                  f"mask_p_range: [{batch_mask_p.min():.4f}, {batch_mask_p.max():.4f}],"
                  f" mask_p_mean: {batch_mask_p.mean():.4f}")
        
        return batch_mask_p, loss_weights
    
    @staticmethod
    def _uniform_schedule(batch_size: int, device: torch.device, 
                        global_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uniform noise schedule."""
        t_values = MaskingStrategies.sample_low_discrepancy(batch_size, device)
        alpha_t = 1.0 - t_values
        batch_mask_p = 1.0 - alpha_t
        derivatives = -torch.ones_like(t_values)
        loss_weights = torch.abs(derivatives) / (1 - alpha_t + 1e-8)
        
        if global_step % 200 == 0:
            print(f"Uniform schedule at step {global_step}, "
                  f"mask_p_range: [{batch_mask_p.min():.4f}, {batch_mask_p.max():.4f}]")
        
        return batch_mask_p, loss_weights
    
    @staticmethod
    def _gaussian_schedule(args: argparse.Namespace, batch_size: int, device: torch.device, 
                         global_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gaussian noise schedule."""
        t_values = MaskingStrategies.sample_low_discrepancy(batch_size, device)
        
        mu = getattr(args, 'gaussian_mu', 0.2)
        sigma = getattr(args, 'gaussian_sigma', 0.12)
        
        t_clamped = torch.clamp(t_values, 1e-8, 1-1e-8)
        z = math.sqrt(2) * torch.erfinv(2 * t_clamped - 1)
        alpha_t = (1 - mu) + sigma * z
        alpha_t = torch.clamp(alpha_t, 1e-8, 1-1e-8)
        batch_mask_p = 1.0 - alpha_t
        
        phi_z = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
        derivatives = sigma / phi_z
        
        gaussian_power = getattr(args, 'gaussian_power', 1.0)
        loss_weights = torch.abs(derivatives) ** gaussian_power / (1 - alpha_t + 1e-8)
        
        if global_step % 200 == 0:
            print(f"Gaussian schedule at step {global_step}, "
                  f"mask_p_range: [{batch_mask_p.min():.4f}, {batch_mask_p.max():.4f}], "
                  f"mask_p_mean: {batch_mask_p.mean():.4f} (target: {mu:.4f})")
        
        return batch_mask_p, loss_weights
    
    @staticmethod
    def _bimodal_gaussian_schedule(args: argparse.Namespace, batch_size: int, 
                                 device: torch.device, global_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bimodal Gaussian noise schedule."""
        t_values = MaskingStrategies.sample_low_discrepancy(batch_size, device)
        
        # Get bimodal parameters
        left_mean = getattr(args, 'bimodal_left_mean', 0.12)
        left_std = getattr(args, 'bimodal_left_std', 0.02)
        right_initial = getattr(args, 'bimodal_right_initial', 0.2)
        right_final = getattr(args, 'bimodal_right_final', 0.85)
        exp_rate = getattr(args, 'bimodal_exp_rate', 1.0)
        weight_left = getattr(args, 'bimodal_weight_left', 0.6)
        right_std = getattr(args, 'bimodal_right_std', 0.08)
        max_steps = getattr(args, 'max_steps', 100000)
        
        # Calculate dynamic right mean
        progress = min(global_step / max_steps, 1.0)
        exp_progress = 1.0 - math.exp(-exp_rate * progress)
        right_mean = right_initial + (right_final - right_initial) * exp_progress
        
        t_clamped = torch.clamp(t_values, 1e-8, 1 - 1e-8)
        component_choice = (torch.rand(batch_size, device=device) < weight_left).float()
        
        # Left and right components
        z_left = math.sqrt(2) * torch.erfinv(2 * t_clamped - 1)
        alpha_left_t = torch.clamp((1 - left_mean) + left_std * z_left, 1e-8, 1 - 1e-8)
        
        z_right = math.sqrt(2) * torch.erfinv(2 * t_clamped - 1)
        alpha_right_t = torch.clamp((1 - right_mean) + right_std * z_right, 1e-8, 1 - 1e-8)
        
        alpha_t = component_choice * alpha_left_t + (1 - component_choice) * alpha_right_t
        batch_mask_p = 1.0 - alpha_t
        
        # Derivatives
        derivatives_left = left_std * math.sqrt(2 * math.pi) * torch.exp(0.5 * z_left**2)
        derivatives_right = right_std * math.sqrt(2 * math.pi) * torch.exp(0.5 * z_right**2)
        derivatives = component_choice * derivatives_left + (1 - component_choice) * derivatives_right
        
        gaussian_power = getattr(args, 'gaussian_power', 1.0)
        loss_weights = torch.abs(derivatives) ** gaussian_power / (1 - alpha_t + 1e-8)
        
        if global_step % 200 == 0:
            actual_mean = batch_mask_p.mean().item()
            expected_mean = weight_left * left_mean + (1 - weight_left) * right_mean
            print(f"Bimodal Gaussian schedule at step {global_step}, "
                  f"progress: {progress:.3f}, right_mean: {right_mean:.3f}, "
                  f"mask_p_range: [{batch_mask_p.min():.4f}, {batch_mask_p.max():.4f}], "
                  f"mask_p_mean: {actual_mean:.4f} (expected: {expected_mean:.4f})")
        
        return batch_mask_p, loss_weights
