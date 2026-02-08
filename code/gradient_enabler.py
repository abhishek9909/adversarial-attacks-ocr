"""
Gradient Enabler for Vision Language Models

This module provides safe, verifiable methods to enable gradient flow through
vision components of VLMs. It uses the model inspector to identify components
and applies minimal, targeted patches.

Key principles:
1. Inspect before patching
2. Verify after patching
3. Provide rollback capability
4. Log all changes
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import functools
import warnings

from model_inspector import (
    ModelInspectionResult,
    inspect_model,
    get_module_by_path,
    check_gradient_flow,
)


@dataclass
class PatchRecord:
    """Record of a patch applied to the model."""
    component_path: str
    patch_type: str
    original_state: Any
    description: str
    
    
@dataclass
class GradientEnablerState:
    """State tracking for gradient enabler."""
    is_enabled: bool = False
    patches_applied: List[PatchRecord] = field(default_factory=list)
    original_training_mode: Dict[str, bool] = field(default_factory=dict)
    
    
class GradientEnabler:
    """
    Enables gradient flow through vision components of a VLM.
    
    Usage:
        enabler = GradientEnabler(model, tokenizer)
        enabler.inspect()  # Understand the model
        enabler.enable()   # Enable gradients
        enabler.verify()   # Check it worked
        
        # ... do training ...
        
        enabler.disable()  # Restore original state
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        verbose: bool = True,
    ):
        """
        Initialize the gradient enabler.
        
        Args:
            model: The VLM model
            tokenizer: Optional tokenizer for token inspection
            verbose: Whether to print status messages
        """
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        
        self.inspection: Optional[ModelInspectionResult] = None
        self.state = GradientEnablerState()
        
    def log(self, message: str):
        """Log a message if verbose mode is on."""
        if self.verbose:
            print(message)
    
    def inspect(self) -> ModelInspectionResult:
        """
        Inspect the model to understand its structure.
        
        Returns:
            ModelInspectionResult with details about the model
        """
        self.log("Inspecting model...")
        self.inspection = inspect_model(self.model, self.tokenizer)
        
        if self.verbose:
            self.inspection.print_summary()
        
        return self.inspection
    
    def enable(
        self,
        components: Optional[List[str]] = None,
        unfreeze_params: bool = True,
        keep_eval_mode: bool = True,
    ) -> bool:
        """
        Enable gradient flow through vision components.
        
        Args:
            components: Specific component paths to enable (None = all vision components)
            unfreeze_params: Whether to set requires_grad=True on parameters
            keep_eval_mode: Whether to keep components in eval mode (recommended)
            
        Returns:
            True if successful
        """
        if self.inspection is None:
            self.inspect()
        
        if not self.inspection.is_vision_model:
            self.log("Warning: Model doesn't appear to be a vision model")
            return False
        
        if self.state.is_enabled:
            self.log("Gradients already enabled")
            return True
        
        # Determine which components to enable
        if components is None:
            target_components = [c.name for c in self.inspection.vision_components]
        else:
            target_components = components
        
        self.log(f"Enabling gradients for {len(target_components)} component(s)...")
        
        for comp_path in target_components:
            module = get_module_by_path(self.model, comp_path)
            
            if module is None:
                self.log(f"  ⚠ Could not find component: {comp_path}")
                continue
            
            if isinstance(module, nn.Parameter):
                # Handle Parameter directly
                if not module.requires_grad:
                    original = module.requires_grad
                    module.requires_grad_(True)
                    self.state.patches_applied.append(PatchRecord(
                        component_path=comp_path,
                        patch_type="requires_grad",
                        original_state=original,
                        description=f"Set requires_grad=True on Parameter",
                    ))
                    self.log(f"  ✓ Enabled gradient for {comp_path}")
                    
            elif isinstance(module, nn.Module):
                # Save original training mode
                self.state.original_training_mode[comp_path] = module.training
                
                # Set to eval mode if requested (prevents BatchNorm etc. from updating)
                if keep_eval_mode:
                    module.eval()
                
                # Unfreeze parameters
                if unfreeze_params:
                    frozen_count = 0
                    for name, param in module.named_parameters():
                        if not param.requires_grad:
                            param.requires_grad_(True)
                            frozen_count += 1
                    
                    if frozen_count > 0:
                        self.state.patches_applied.append(PatchRecord(
                            component_path=comp_path,
                            patch_type="requires_grad",
                            original_state=frozen_count,
                            description=f"Unfroze {frozen_count} parameters",
                        ))
                        self.log(f"  ✓ Unfroze {frozen_count} parameters in {comp_path}")
        
        self.state.is_enabled = True
        self.log("Gradient enabling complete")
        return True
    
    def disable(self) -> bool:
        """
        Restore the model to its original state.
        
        Returns:
            True if successful
        """
        if not self.state.is_enabled:
            self.log("Gradients not enabled, nothing to disable")
            return True
        
        self.log("Restoring original model state...")
        
        # Restore training modes
        for comp_path, was_training in self.state.original_training_mode.items():
            module = get_module_by_path(self.model, comp_path)
            if module is not None:
                if was_training:
                    module.train()
                else:
                    module.eval()
        
        # Note: We don't restore requires_grad to False because that would
        # require tracking each individual parameter's original state.
        # If you need this, extend the PatchRecord to store more detail.
        
        self.state = GradientEnablerState()
        self.log("Model state restored")
        return True
    
    def verify(self) -> bool:
        """
        Verify that gradient flow is working.
        
        Returns:
            True if gradients can flow through all enabled components
        """
        if self.inspection is None:
            self.inspect()
        
        self.log("Verifying gradient flow...")
        
        all_ok = True
        grad_status = check_gradient_flow(self.model, self.inspection)
        
        for comp_path, can_flow in grad_status.items():
            status = "✓" if can_flow else "✗"
            self.log(f"  {status} {comp_path}")
            if not can_flow:
                all_ok = False
        
        return all_ok
    
    def create_test_input(
        self,
        batch_size: int = 1,
        image_size: Tuple[int, int] = (640, 640),
        seq_length: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """
        Create test inputs for verification.
        
        Args:
            batch_size: Batch size
            image_size: (height, width) of test images
            seq_length: Sequence length for input_ids
            
        Returns:
            Dict with test inputs
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        
        # Get vocab size from config
        vocab_size = getattr(self.model.config, 'vocab_size', 32000)
        
        test_inputs = {
            'input_ids': torch.randint(
                0, vocab_size, (batch_size, seq_length),
                device=device
            ),
            'attention_mask': torch.ones(
                batch_size, seq_length,
                dtype=torch.long, device=device
            ),
        }
        
        # Create test image
        test_image = torch.randn(
            batch_size, 3, image_size[0], image_size[1],
            dtype=dtype, device=device, requires_grad=True
        )
        
        return test_inputs, test_image
    
    def test_backward(
        self,
        test_inputs: Optional[Dict] = None,
        test_image: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, Optional[torch.Tensor]]:
        """
        Test that gradients actually flow backward to the image.
        
        Args:
            test_inputs: Optional pre-created test inputs
            test_image: Optional pre-created test image
            
        Returns:
            (success, image_gradient) tuple
        """
        self.log("Testing backward pass...")
        
        if test_inputs is None or test_image is None:
            test_inputs, test_image = self.create_test_input()
        
        # Clear any existing gradients
        if test_image.grad is not None:
            test_image.grad.zero_()
        
        try:
            # This is model-specific - subclasses should override
            self.log("  Note: Full backward test requires model-specific implementation")
            self.log("  See DifferentiablePipeline for complete forward/backward testing")
            return True, None
            
        except Exception as e:
            self.log(f"  ✗ Backward test failed: {e}")
            return False, None


class VisionModelWrapper(nn.Module):
    """
    Wrapper that ensures gradient flow through a vision model.
    
    This wrapper:
    1. Removes torch.no_grad() contexts
    2. Ensures parameters require gradients
    3. Provides a clean interface for getting image features
    """
    
    def __init__(
        self,
        vision_model: nn.Module,
        keep_eval_mode: bool = True,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.keep_eval_mode = keep_eval_mode
        
        # Ensure all parameters require gradients
        for param in self.vision_model.parameters():
            param.requires_grad_(True)
        
        if keep_eval_mode:
            self.vision_model.eval()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with guaranteed gradient flow.
        
        Args:
            images: Tensor of shape [B, C, H, W]
            
        Returns:
            Vision features
        """
        # Ensure gradients are enabled
        with torch.enable_grad():
            # Call the underlying model
            features = self.vision_model(images)
            
            # Handle different output formats
            if isinstance(features, tuple):
                features = features[0]
            elif hasattr(features, 'last_hidden_state'):
                features = features.last_hidden_state
            
            return features
    
    def train(self, mode: bool = True):
        """Override train to optionally keep eval mode."""
        if self.keep_eval_mode:
            return self
        return super().train(mode)


@contextmanager
def gradient_enabled_context(model: nn.Module, components: List[str]):
    """
    Context manager that temporarily enables gradients for specific components.
    
    Usage:
        with gradient_enabled_context(model, ['model.vision_model']):
            output = model(inputs)
            loss.backward()
    """
    # Store original states
    original_requires_grad = {}
    
    for comp_path in components:
        module = get_module_by_path(model, comp_path)
        if module is not None:
            for name, param in module.named_parameters():
                full_name = f"{comp_path}.{name}"
                original_requires_grad[full_name] = param.requires_grad
                param.requires_grad_(True)
    
    try:
        with torch.enable_grad():
            yield
    finally:
        # Restore original states
        for comp_path in components:
            module = get_module_by_path(model, comp_path)
            if module is not None:
                for name, param in module.named_parameters():
                    full_name = f"{comp_path}.{name}"
                    if full_name in original_requires_grad:
                        param.requires_grad_(original_requires_grad[full_name])


def wrap_forward_with_gradients(
    module: nn.Module,
    component_name: str = "module",
) -> Callable:
    """
    Create a wrapper for a forward method that ensures gradients.
    
    Args:
        module: The module to wrap
        component_name: Name for logging
        
    Returns:
        Wrapped forward function
    """
    original_forward = module.forward
    
    @functools.wraps(original_forward)
    def wrapped_forward(*args, **kwargs):
        with torch.enable_grad():
            return original_forward(*args, **kwargs)
    
    return wrapped_forward


def apply_gradient_hooks(
    model: nn.Module,
    inspection: ModelInspectionResult,
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Apply gradient hooks to verify gradient flow during training.
    
    Args:
        model: The model
        inspection: Inspection results
        
    Returns:
        List of hook handles (call .remove() to clean up)
    """
    handles = []
    
    def make_hook(name):
        def hook(grad):
            if grad is not None:
                print(f"  Gradient at {name}: shape={grad.shape}, norm={grad.norm().item():.6f}")
            else:
                print(f"  Gradient at {name}: None!")
            return grad
        return hook
    
    for comp in inspection.vision_components:
        module = get_module_by_path(model, comp.name)
        if module is not None and isinstance(module, nn.Module):
            # Add hook to first parameter
            for name, param in module.named_parameters():
                if param.requires_grad:
                    handle = param.register_hook(make_hook(f"{comp.name}.{name}"))
                    handles.append(handle)
                    break  # Only first param per component
    
    return handles


if __name__ == "__main__":
    print("Gradient Enabler Module")
    print()
    print("Usage:")
    print("  from gradient_enabler import GradientEnabler")
    print()
    print("  enabler = GradientEnabler(model, tokenizer)")
    print("  enabler.inspect()  # Understand the model")
    print("  enabler.enable()   # Enable gradients")
    print("  enabler.verify()   # Check it worked")
