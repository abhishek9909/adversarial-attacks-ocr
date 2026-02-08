"""
Differentiable DeepSeek OCR Pipeline

Main entry point for creating an end-to-end differentiable pipeline.
This module orchestrates model loading, gradient enabling, and provides
a clean API for training with image gradients.

Usage:
    from differentiable_pipeline import DifferentiablePipeline
    
    pipeline = DifferentiablePipeline.from_pretrained(
        "deepseek-ai/deepseek-vl2-tiny",
        enable_image_gradients=True,
    )
    
    # Training
    output = pipeline.forward(input_ids, images, labels=labels)
    output.loss.backward()
    
    # Get image gradients
    image_grads = pipeline.get_image_gradients()
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

from model_inspector import inspect_model, ModelInspectionResult, get_module_by_path
from gradient_enabler import GradientEnabler
from differentiable_transforms import (
    DifferentiableImageProcessor,
    pil_to_tensor,
    verify_gradient_flow,
)
from transformers import AutoModel

@dataclass
class DifferentiableOutput:
    """Output from differentiable forward pass."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    image_features: Optional[torch.Tensor] = None
    
    def __getitem__(self, key):
        return getattr(self, key)


class DifferentiablePipeline:
    """
    End-to-end differentiable pipeline for vision-language models.
    
    This class:
    1. Loads the model (optionally with Unsloth)
    2. Inspects and patches for gradient flow
    3. Provides differentiable image preprocessing
    4. Manages forward pass with proper gradient tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        image_processor: Optional[DifferentiableImageProcessor] = None,
        enable_image_gradients: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize the pipeline with an existing model.
        
        Args:
            model: Pre-loaded VLM model
            tokenizer: Tokenizer
            image_processor: Optional custom image processor
            enable_image_gradients: Whether to enable gradient flow
            verbose: Whether to print status messages
        """
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.enable_image_gradients = enable_image_gradients
        
        # Initialize image processor
        self.image_processor = image_processor or DifferentiableImageProcessor()
        
        # Initialize gradient enabler
        self.gradient_enabler = GradientEnabler(model, tokenizer, verbose=verbose)
        
        # Track current images for gradient retrieval
        self._current_images: Optional[torch.Tensor] = None
        self._inspection: Optional[ModelInspectionResult] = None
        
        # Setup if gradients enabled
        if enable_image_gradients:
            self._setup_gradients()
    
    def _setup_gradients(self):
        """Setup gradient flow through vision components."""
        self._inspection = self.gradient_enabler.inspect()
        
        if not self._inspection.is_vision_model:
            warnings.warn(
                "Model doesn't appear to have vision components. "
                "Gradient enabling may not work as expected."
            )
        
        self.gradient_enabler.enable(
            unfreeze_params=True,
            keep_eval_mode=True,
        )
        
        success = self.gradient_enabler.verify()
        if not success:
            warnings.warn("Gradient verification failed for some components")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        enable_image_gradients: bool = True,
        use_unsloth: bool = False,
        load_in_4bit: bool = False,
        torch_dtype=None,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> "DifferentiablePipeline":
        """
        Load a model and create a differentiable pipeline.
        
        Args:
            model_name_or_path: Model identifier or path
            enable_image_gradients: Enable gradient flow through images
            use_unsloth: Use Unsloth for optimized loading
            load_in_4bit: Load in 4-bit quantization
            torch_dtype: Data type for model weights
            device_map: Device mapping strategy
            trust_remote_code: Allow remote code execution
            verbose: Print status messages
            **kwargs: Additional arguments for model loading
            
        Returns:
            DifferentiablePipeline instance
        """
        if verbose:
            print(f"Loading model: {model_name_or_path}")
            print(f"  Image gradients: {'ENABLED' if enable_image_gradients else 'DISABLED'}")
            print(f"  Using Unsloth: {use_unsloth}")
            print(f"  4-bit quantization: {load_in_4bit}")
        
        if use_unsloth:
            model, tokenizer = cls._load_with_unsloth(
                model_name_or_path,
                load_in_4bit=load_in_4bit,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        else:
            model, tokenizer = cls._load_standard(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            enable_image_gradients=enable_image_gradients,
            verbose=verbose,
        )
    
    @staticmethod
    def _load_with_unsloth(
        model_name: str,
        load_in_4bit: bool = True,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> Tuple[nn.Module, Any]:
        """Load model using Unsloth."""
        try:
            from unsloth import FastVisionModel
            import os
            os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'
        except ImportError:
            raise ImportError(
                "Unsloth is required for use_unsloth=True. "
                "Install with: pip install unsloth"
            )
        
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
            auto_model=AutoModel,
            unsloth_force_compile=True,
            **kwargs,
        )
        
        return model, tokenizer
    
    @staticmethod
    def _load_standard(
        model_name: str,
        torch_dtype=None,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        **kwargs,
    ) -> Tuple[nn.Module, Any]:
        """Load model using standard transformers."""
        from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
        
        # Try to load with AutoProcessor first (handles image processing)
        try:
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
        
        # Load model
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        
        return model, tokenizer
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.model.parameters()).device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model."""
        return next(self.model.parameters()).dtype
    
    def prepare_image(
        self,
        image,
        requires_grad: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Prepare an image for the model.
        
        Args:
            image: PIL Image or torch.Tensor
            requires_grad: Override gradient requirement (None = use pipeline setting)
            
        Returns:
            (local_patches, global_view, info) tuple
        """
        if requires_grad is None:
            requires_grad = self.enable_image_gradients
        
        # Convert PIL to tensor if needed
        if not isinstance(image, torch.Tensor):
            image = pil_to_tensor(
                image,
                device=self.device,
                dtype=self.dtype,
                requires_grad=requires_grad,
            )
        elif requires_grad and not image.requires_grad:
            image = image.requires_grad_(True)
        
        # Store for gradient retrieval
        if requires_grad:
            self._current_images = image
        
        # Process through differentiable transforms
        local_patches, global_view, info = self.image_processor(
            image, return_info=True
        )
        
        return local_patches, global_view, info
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[Union[torch.Tensor, List]] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> DifferentiableOutput:
        """
        Forward pass with gradient support.
        
        This is a generic forward pass. For model-specific handling,
        use the model's forward method directly or subclass this.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            images: Images (tensor or list of PIL images)
            labels: Labels for loss computation
            **kwargs: Additional model-specific arguments
            
        Returns:
            DifferentiableOutput with loss, logits, etc.
        """
        # Prepare inputs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Process images if provided
        if images is not None:
            if isinstance(images, list):
                # Process each image
                processed_images = []
                for img in images:
                    local, glob, _ = self.prepare_image(img)
                    processed_images.append((local, glob))
                # This is model-specific - may need customization
                kwargs['pixel_values'] = torch.cat(
                    [g for _, g in processed_images], dim=0
                )
            elif isinstance(images, torch.Tensor):
                if self.enable_image_gradients and not images.requires_grad:
                    images = images.requires_grad_(True)
                self._current_images = images
                kwargs['pixel_values'] = images
        
        # Ensure gradients are enabled during forward
        with torch.enable_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        
        # Extract outputs
        return DifferentiableOutput(
            loss=getattr(outputs, 'loss', None),
            logits=getattr(outputs, 'logits', None),
            hidden_states=getattr(outputs, 'hidden_states', None),
            attentions=getattr(outputs, 'attentions', None),
        )
    
    def get_image_gradients(self) -> Optional[torch.Tensor]:
        """
        Get gradients with respect to the input images.
        
        Call this after backward() to get the image gradients.
        
        Returns:
            Image gradients or None if not available
        """
        if self._current_images is None:
            return None
        
        return self._current_images.grad
    
    def compute_image_attribution(
        self,
        input_ids: torch.LongTensor,
        images: torch.Tensor,
        target_token_idx: int = -1,
        target_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute attribution/saliency map for input images.
        
        This shows which parts of the image contribute to the prediction.
        
        Args:
            input_ids: Token IDs
            images: Image tensor (requires_grad will be enabled)
            target_token_idx: Position to compute gradient for (-1 = last)
            target_token_id: Specific token ID to compute gradient for
            **kwargs: Additional forward arguments
            
        Returns:
            Gradient tensor showing image attribution
        """
        # Ensure image has gradients
        if not images.requires_grad:
            images = images.requires_grad_(True)
        
        self._current_images = images
        
        # Forward pass
        output = self.forward(input_ids, images=images, **kwargs)
        
        # Get target logit
        logits = output.logits
        
        if target_token_id is not None:
            target = logits[0, target_token_idx, target_token_id]
        else:
            target = logits[0, target_token_idx].max()
        
        # Backward
        target.backward()
        
        # Get and process gradients
        grads = self.get_image_gradients()
        
        if grads is not None:
            # Compute saliency (absolute gradient)
            saliency = grads.abs()
            # Sum over channels
            if saliency.dim() == 4:
                saliency = saliency.sum(dim=1)
            elif saliency.dim() == 3:
                saliency = saliency.sum(dim=0)
        else:
            saliency = None
        
        return saliency
    
    def verify_gradient_flow(self) -> bool:
        """
        Run a full verification of gradient flow.
        
        Creates test inputs and verifies gradients flow from
        loss back to image inputs.
        
        Returns:
            True if gradients flow correctly
        """
        print("=" * 50)
        print("GRADIENT FLOW VERIFICATION")
        print("=" * 50)
        
        # Create test image
        test_image = torch.randn(
            1, 3, 640, 640,
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        
        # Create test tokens
        test_ids = torch.randint(
            0, self.tokenizer.vocab_size,
            (1, 50),
            device=self.device,
        )
        
        print("\n1. Checking image tensor...")
        verify_gradient_flow(test_image, "test_image")
        
        print("\n2. Checking component gradients...")
        self.gradient_enabler.verify()
        
        print("\n3. Testing backward pass...")
        try:
            # Store image
            self._current_images = test_image
            
            # Forward (this is model-specific and may fail)
            with torch.enable_grad():
                # Try different forward approaches
                try:
                    output = self.model(
                        input_ids=test_ids,
                        pixel_values=test_image,
                    )
                except TypeError:
                    # Model may use different argument names
                    output = self.model(
                        input_ids=test_ids,
                        images=test_image,
                    )
            
            # Get logits
            if hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output[0] if isinstance(output, tuple) else output
            
            # Backward
            loss = logits.sum()
            loss.backward()
            
            # Check gradient
            if test_image.grad is not None:
                grad_norm = test_image.grad.norm().item()
                print(f"   ✓ Gradient received! Norm: {grad_norm:.6f}")
                success = True
            else:
                print("   ✗ No gradient received")
                success = False
                
        except Exception as e:
            print(f"   ✗ Forward/backward failed: {e}")
            print("   Note: This may be expected if model has custom forward signature")
            success = False
        
        print("\n" + "=" * 50)
        print(f"RESULT: {'PASS' if success else 'FAIL'}")
        print("=" * 50)
        
        return success
    
    def get_vision_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract vision features from images.
        
        This accesses the vision encoder directly.
        
        Args:
            images: Image tensor
            
        Returns:
            Vision features
        """
        if self._inspection is None:
            self._inspection = self.gradient_enabler.inspect()
        
        # Find vision encoder
        vision_path = self._inspection.vision_encoder_path
        if vision_path is None:
            raise RuntimeError("Could not find vision encoder in model")
        
        vision_encoder = get_module_by_path(self.model, vision_path)
        if vision_encoder is None:
            raise RuntimeError(f"Vision encoder not found at {vision_path}")
        
        # Enable gradients and forward
        if self.enable_image_gradients and not images.requires_grad:
            images = images.requires_grad_(True)
        
        with torch.enable_grad():
            features = vision_encoder(images)
        
        return features


def create_differentiable_collator(
    pipeline: DifferentiablePipeline,
    max_length: int = 2048,
    padding: str = "longest",
    return_tensors: str = "pt",
):
    """
    Create a data collator for training with the pipeline.
    
    Args:
        pipeline: DifferentiablePipeline instance
        max_length: Maximum sequence length
        padding: Padding strategy
        return_tensors: Return type
        
    Returns:
        Collator function
    """
    def collate_fn(examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples."""
        # Extract components
        texts = [ex.get('text', '') for ex in examples]
        images = [ex.get('image') for ex in examples if ex.get('image') is not None]
        
        # Tokenize texts
        text_inputs = pipeline.tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=max_length,
            return_tensors=return_tensors,
        )
        
        # Process images
        processed_images = []
        for img in images:
            local, glob, _ = pipeline.prepare_image(img)
            processed_images.append(glob)
        
        batch = {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
        }
        
        if processed_images:
            batch['pixel_values'] = torch.cat(processed_images, dim=0)
        
        return batch
    
    return collate_fn


if __name__ == "__main__":
    print("Differentiable Pipeline Module")
    print()
    print("Usage:")
    print("  from differentiable_pipeline import DifferentiablePipeline")
    print()
    print("  # Load model")
    print("  pipeline = DifferentiablePipeline.from_pretrained(")
    print("      'deepseek-ai/deepseek-vl2-tiny',")
    print("      enable_image_gradients=True,")
    print("  )")
    print()
    print("  # Verify gradients work")
    print("  pipeline.verify_gradient_flow()")
    print()
    print("  # Use for training")
    print("  output = pipeline.forward(input_ids, images=images, labels=labels)")
    print("  output.loss.backward()")
    print("  image_grads = pipeline.get_image_gradients()")
