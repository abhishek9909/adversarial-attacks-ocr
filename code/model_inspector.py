"""
Model Inspector for DeepSeek OCR

This module inspects the actual model architecture to understand:
1. What vision components exist
2. Where gradients are blocked
3. What needs to be patched

Run this FIRST to understand your model before applying any patches.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class VisionComponentInfo:
    """Information about a vision component in the model."""
    name: str
    module_type: str
    has_forward: bool
    param_count: int
    requires_grad: bool
    device: str
    dtype: str
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "module_type": self.module_type,
            "has_forward": self.has_forward,
            "param_count": self.param_count,
            "requires_grad": self.requires_grad,
            "device": self.device,
            "dtype": self.dtype,
        }


@dataclass
class ModelInspectionResult:
    """Results from inspecting a model."""
    model_class: str
    is_vision_model: bool
    vision_components: List[VisionComponentInfo] = field(default_factory=list)
    vision_encoder_path: Optional[str] = None
    projector_path: Optional[str] = None
    language_model_path: Optional[str] = None
    image_token_id: Optional[int] = None
    special_tokens: Dict[str, int] = field(default_factory=dict)
    config_info: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "model_class": self.model_class,
            "is_vision_model": self.is_vision_model,
            "vision_components": [c.to_dict() for c in self.vision_components],
            "vision_encoder_path": self.vision_encoder_path,
            "projector_path": self.projector_path,
            "language_model_path": self.language_model_path,
            "image_token_id": self.image_token_id,
            "special_tokens": self.special_tokens,
            "config_info": self.config_info,
            "warnings": self.warnings,
        }
    
    def print_summary(self):
        """Print a human-readable summary."""
        print("=" * 60)
        print("MODEL INSPECTION RESULTS")
        print("=" * 60)
        print(f"Model class: {self.model_class}")
        print(f"Is vision model: {self.is_vision_model}")
        print()
        
        if self.vision_components:
            print("Vision Components Found:")
            for comp in self.vision_components:
                grad_status = "✓ gradients" if comp.requires_grad else "✗ frozen"
                print(f"  - {comp.name}: {comp.module_type} ({comp.param_count:,} params, {grad_status})")
        else:
            print("No vision components found!")
        
        print()
        print(f"Vision encoder path: {self.vision_encoder_path or 'Not found'}")
        print(f"Projector path: {self.projector_path or 'Not found'}")
        print(f"Language model path: {self.language_model_path or 'Not found'}")
        print(f"Image token ID: {self.image_token_id or 'Not found'}")
        
        if self.warnings:
            print()
            print("Warnings:")
            for w in self.warnings:
                print(f"  ⚠ {w}")
        
        print("=" * 60)


def inspect_model(model: nn.Module, tokenizer=None) -> ModelInspectionResult:
    """
    Inspect a model to understand its architecture.
    
    Args:
        model: The model to inspect
        tokenizer: Optional tokenizer to find special tokens
        
    Returns:
        ModelInspectionResult with detailed information
    """
    result = ModelInspectionResult(
        model_class=model.__class__.__name__,
        is_vision_model=False,
    )
    
    # Extract config info
    if hasattr(model, 'config'):
        config = model.config
        result.config_info = {
            "model_type": getattr(config, "model_type", "unknown"),
            "hidden_size": getattr(config, "hidden_size", None),
            "vocab_size": getattr(config, "vocab_size", None),
        }
        
        # Check for vision config
        if hasattr(config, "vision_config"):
            result.is_vision_model = True
            result.config_info["vision_config"] = str(type(config.vision_config))
    
    # Common vision component names to look for
    vision_component_names = [
        # DeepSeek OCR specific
        "sam_model", "vision_model", "vision_tower", "visual",
        "projector", "mm_projector", "multi_modal_projector",
        "image_newline", "view_seperator", "view_separator",
        # Generic VLM names
        "vision_encoder", "image_encoder", "visual_encoder",
        "vit", "clip", "siglip", "eva",
    ]
    
    # Find the base model (might be wrapped)
    base_model = model
    if hasattr(model, 'model'):
        base_model = model.model
        result.language_model_path = "model"
    elif hasattr(model, 'language_model'):
        base_model = model.language_model
        result.language_model_path = "language_model"
    
    # Search for vision components
    def search_for_component(obj, path=""):
        """Recursively search for vision components."""
        for name in vision_component_names:
            if hasattr(obj, name):
                component = getattr(obj, name)
                if component is not None and isinstance(component, (nn.Module, nn.Parameter)):
                    full_path = f"{path}.{name}" if path else name
                    
                    # Get component info
                    if isinstance(component, nn.Module):
                        param_count = sum(p.numel() for p in component.parameters())
                        requires_grad = any(p.requires_grad for p in component.parameters())
                        
                        # Get device/dtype from first parameter
                        first_param = next(component.parameters(), None)
                        device = str(first_param.device) if first_param is not None else "unknown"
                        dtype = str(first_param.dtype) if first_param is not None else "unknown"
                        
                        info = VisionComponentInfo(
                            name=full_path,
                            module_type=component.__class__.__name__,
                            has_forward=hasattr(component, 'forward'),
                            param_count=param_count,
                            requires_grad=requires_grad,
                            device=device,
                            dtype=dtype,
                        )
                        result.vision_components.append(info)
                        result.is_vision_model = True
                        
                        # Identify component type
                        name_lower = name.lower()
                        if any(x in name_lower for x in ["sam", "vision", "visual", "encoder", "tower"]):
                            if result.vision_encoder_path is None:
                                result.vision_encoder_path = full_path
                        elif "projector" in name_lower:
                            result.projector_path = full_path
                    
                    elif isinstance(component, nn.Parameter):
                        info = VisionComponentInfo(
                            name=full_path,
                            module_type="Parameter",
                            has_forward=False,
                            param_count=component.numel(),
                            requires_grad=component.requires_grad,
                            device=str(component.device),
                            dtype=str(component.dtype),
                        )
                        result.vision_components.append(info)
    
    # Search in model and base_model
    search_for_component(model)
    if base_model is not model:
        search_for_component(base_model, "model")
    
    # Look for image token ID
    if tokenizer is not None:
        # Common image token patterns
        image_token_patterns = ["<image>", "<|image|>", "[IMG]", "<img>"]
        for pattern in image_token_patterns:
            token_id = tokenizer.convert_tokens_to_ids(pattern)
            if token_id != tokenizer.unk_token_id:
                result.image_token_id = token_id
                result.special_tokens[pattern] = token_id
                break
        
        # Also check vocab directly for image-related tokens
        if hasattr(tokenizer, 'get_vocab'):
            vocab = tokenizer.get_vocab()
            for token, id in vocab.items():
                if 'image' in token.lower() and id > 100000:  # Likely special token
                    result.special_tokens[token] = id
                    if result.image_token_id is None:
                        result.image_token_id = id
    
    # Add warnings
    if not result.vision_components:
        result.warnings.append("No vision components found - model may not be a VLM")
    
    frozen_components = [c for c in result.vision_components if not c.requires_grad]
    if frozen_components:
        result.warnings.append(
            f"{len(frozen_components)} vision component(s) have frozen gradients: "
            f"{[c.name for c in frozen_components]}"
        )
    
    return result


def get_module_by_path(model: nn.Module, path: str) -> Optional[nn.Module]:
    """
    Get a module by its dot-separated path.
    
    Args:
        model: Root model
        path: Dot-separated path like "model.vision_model"
        
    Returns:
        The module at that path, or None if not found
    """
    parts = path.split('.')
    current = model
    
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    
    return current


def check_gradient_flow(model: nn.Module, inspection: ModelInspectionResult) -> Dict[str, bool]:
    """
    Check if gradients can flow through vision components.
    
    Args:
        model: The model
        inspection: Previous inspection results
        
    Returns:
        Dict mapping component paths to whether gradients can flow
    """
    results = {}
    
    for comp in inspection.vision_components:
        module = get_module_by_path(model, comp.name)
        if module is None:
            results[comp.name] = False
            continue
        
        if isinstance(module, nn.Parameter):
            results[comp.name] = module.requires_grad
        elif isinstance(module, nn.Module):
            # Check if any parameters require grad
            has_grad_params = any(p.requires_grad for p in module.parameters())
            results[comp.name] = has_grad_params
        else:
            results[comp.name] = False
    
    return results


def print_model_tree(model: nn.Module, max_depth: int = 3, show_params: bool = False):
    """
    Print a tree view of the model structure.
    
    Args:
        model: The model to visualize
        max_depth: Maximum depth to show
        show_params: Whether to show parameter shapes
    """
    def _print_tree(module, prefix="", depth=0):
        if depth > max_depth:
            return
        
        # Get children
        children = list(module.named_children())
        
        for i, (name, child) in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            
            # Get param count
            param_count = sum(p.numel() for p in child.parameters(recurse=False))
            param_str = f" ({param_count:,} params)" if param_count > 0 else ""
            
            print(f"{prefix}{connector}{name}: {child.__class__.__name__}{param_str}")
            
            if show_params:
                # Show direct parameters
                for pname, param in child.named_parameters(recurse=False):
                    param_prefix = prefix + ("    " if is_last else "│   ")
                    grad_str = "✓" if param.requires_grad else "✗"
                    print(f"{param_prefix}[{grad_str}] {pname}: {list(param.shape)}")
            
            # Recurse
            new_prefix = prefix + ("    " if is_last else "│   ")
            _print_tree(child, new_prefix, depth + 1)
    
    print(f"Model: {model.__class__.__name__}")
    _print_tree(model)


if __name__ == "__main__":
    # Example usage
    print("Model Inspector Module")
    print("Usage:")
    print("  from model_inspector import inspect_model, print_model_tree")
    print("  result = inspect_model(model, tokenizer)")
    print("  result.print_summary()")
    print("  print_model_tree(model)")
