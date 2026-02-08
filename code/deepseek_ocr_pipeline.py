"""
DeepSeek OCR Specific Implementation

This module provides DeepSeek-specific handling for the differentiable pipeline.
It understands DeepSeek's unique architecture including:
- SAM model for local features
- Vision model for global features  
- Projector for feature mapping
- Custom forward signature

This is designed to work with deepseek-ai/deepseek-vl2 family models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings

from differentiable_pipeline import DifferentiablePipeline, DifferentiableOutput
from differentiable_transforms import (
    DifferentiableImageProcessor,
    pil_to_tensor
)
from model_inspector import inspect_model, get_module_by_path


@dataclass
class DeepSeekVisionComponents:
    """Holds references to DeepSeek vision components."""
    sam_model: Optional[nn.Module] = None
    vision_model: Optional[nn.Module] = None
    projector: Optional[nn.Module] = None
    image_newline: Optional[nn.Parameter] = None
    view_separator: Optional[nn.Parameter] = None
    
    def is_complete(self) -> bool:
        """Check if all required components are found."""
        return (
            self.sam_model is not None and
            self.vision_model is not None and
            self.projector is not None
        )


class DeepSeekOCRPipeline(DifferentiablePipeline):
    """
    Differentiable pipeline specifically for DeepSeek OCR/VL models.
    
    This handles the specific architecture of DeepSeek models:
    - Dual vision encoder (SAM + ViT)
    - Custom image token handling
    - Specific forward signature
    """
    
    # Known DeepSeek model configurations
    KNOWN_CONFIGS = {
        "deepseek-ocr": {
            "image_token": "<image>",
            "patch_size": 640,
            "base_size": 1024,
            "downsample_ratio": 4,
        },
        "deepseek-vl2": {
            "image_token": "<image>",
            "patch_size": 640,
            "base_size": 1024,
            "downsample_ratio": 4,
        },
    }
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        enable_image_gradients: bool = True,
        verbose: bool = True,
        config_override: Optional[Dict] = None,
    ):
        """
        Initialize DeepSeek-specific pipeline.
        
        Args:
            model: DeepSeek model
            tokenizer: Tokenizer
            enable_image_gradients: Enable gradient flow
            verbose: Print status
            config_override: Override default configuration
        """
        # Store model and tokenizer early (needed by helper methods)
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        
        # Get configuration
        self.ds_config = self._detect_config(model, config_override)
        
        # Find vision components BEFORE calling super().__init__
        # (because _setup_gradients needs them)
        self.vision_components = self._find_vision_components()
        
        # Get image token ID
        self.image_token_id = self._find_image_token_id()
        
        # Create DeepSeek-specific image processor
        image_processor = DeepSeekImageProcessor(
            patch_size=self.ds_config["patch_size"],
            base_size=self.ds_config["base_size"],
        )
        
        # Initialize base class (this will call _setup_gradients)
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            enable_image_gradients=enable_image_gradients,
            verbose=verbose,
        )
        
        if verbose:
            self._print_component_status()
    
    def _detect_config(
        self, 
        model: nn.Module, 
        override: Optional[Dict]
    ) -> Dict:
        """Detect model configuration."""
        config = {
            "image_token": "<image>",
            "patch_size": 640,
            "base_size": 1024,
            "downsample_ratio": 4,
        }
        
        # Try to detect from model config
        if hasattr(model, 'config'):
            mc = model.config
            if hasattr(mc, 'vision_config'):
                vc = mc.vision_config
                if hasattr(vc, 'image_size'):
                    config["patch_size"] = vc.image_size
        
        # Apply overrides
        if override:
            config.update(override)
        
        return config
    
    def _find_vision_components(self) -> DeepSeekVisionComponents:
        """Find DeepSeek-specific vision components."""
        components = DeepSeekVisionComponents()
        
        # Get base model
        base = self.model.model if hasattr(self.model, 'model') else self.model
        
        # Look for components
        component_names = [
            ('sam_model', 'sam_model'),
            ('vision_model', 'vision_model'),
            ('projector', 'projector'),
            ('image_newline', 'image_newline'),
            ('view_separator', 'view_seperator'),  # Note: typo in original
            ('view_separator', 'view_separator'),
        ]
        
        for attr_name, search_name in component_names:
            if hasattr(base, search_name):
                setattr(components, attr_name, getattr(base, search_name))
            elif hasattr(self.model, search_name):
                setattr(components, attr_name, getattr(self.model, search_name))
        
        return components
    
    def _find_image_token_id(self) -> int:
        """Find the image token ID in the tokenizer."""
        # Try common patterns
        patterns = ["<image>", "<|image|>", "[IMG]", "<img>"]
        
        for pattern in patterns:
            token_id = self.tokenizer.convert_tokens_to_ids(pattern)
            if token_id != self.tokenizer.unk_token_id:
                return token_id
        
        # Check vocab for image-related high-ID tokens
        if hasattr(self.tokenizer, 'get_vocab'):
            vocab = self.tokenizer.get_vocab()
            for token, tid in vocab.items():
                if 'image' in token.lower() and tid > 100000:
                    return tid
        
        # Fallback (DeepSeek OCR default)
        warnings.warn("Could not find image token, using default 128815")
        return 128815
    
    def _print_component_status(self):
        """Print status of found components."""
        print("\nDeepSeek Vision Components:")
        print(f"  SAM model: {'✓ Found' if self.vision_components.sam_model is not None else '✗ Missing'}")
        print(f"  Vision model: {'✓ Found' if self.vision_components.vision_model is not None else '✗ Missing'}")
        print(f"  Projector: {'✓ Found' if self.vision_components.projector is not None else '✗ Missing'}")
        print(f"  Image newline: {'✓ Found' if self.vision_components.image_newline is not None else '✗ Missing'}")
        print(f"  View separator: {'✓ Found' if self.vision_components.view_separator is not None else '✗ Missing'}")
        print(f"  Image token ID: {self.image_token_id}")
        print()
    
    def _setup_gradients(self):
        """Setup gradients for DeepSeek vision components."""
        super()._setup_gradients()
        
        # Additional DeepSeek-specific setup
        if not self.vision_components.is_complete():
            warnings.warn(
                "Some DeepSeek vision components not found. "
                "Gradient flow may not work correctly."
            )
            return
        
        # Enable gradients on vision components
        for name in ['sam_model', 'vision_model', 'projector']:
            component = getattr(self.vision_components, name)
            if component is not None:
                component.eval()  # Keep in eval mode
                for param in component.parameters():
                    param.requires_grad_(True)
                if self.verbose:
                    print(f"  ✓ Enabled gradients on {name}")
        
        # Try to disable donated buffer issue for torch.compile
        self._disable_donated_buffer()
    
    def _disable_donated_buffer(self):
        """
        Disable torch.compile donated buffer to allow backward() calls.
        
        This is needed when using Unsloth or other torch.compile optimizations
        that conflict with gradient computation.
        """
        try:
            from torch._functorch import config as functorch_config
            if hasattr(functorch_config, 'donated_buffer'):
                functorch_config.donated_buffer = False
                if self.verbose:
                    print("  ✓ Disabled torch.compile donated_buffer for gradient support")
        except (ImportError, AttributeError):
            pass  # Not available in this PyTorch version
    
    def process_vision_features(
        self,
        patches: torch.Tensor,
        global_view: torch.Tensor,
        crop_shape: Tuple[int, int] = (1, 1),
    ) -> torch.Tensor:
        """
        Process images through DeepSeek vision encoders.
        
        This replicates DeepSeek's vision processing with gradient support.
        
        Args:
            patches: Local crop patches [N, C, H, W]
            global_view: Global view [1, C, H, W]
            crop_shape: (width_crops, height_crops)
            
        Returns:
            Combined vision features
        """
        if not self.vision_components.is_complete():
            raise RuntimeError("Vision components not fully initialized")
        
        sam = self.vision_components.sam_model
        vision = self.vision_components.vision_model
        proj = self.vision_components.projector
        newline = self.vision_components.image_newline
        separator = self.vision_components.view_separator
        
        # Process with gradient support
        with torch.enable_grad():
            # Check if we have local patches
            has_patches = patches.sum().item() != 0
            
            if has_patches:
                # Local features from patches
                local_feat_1 = sam(patches)
                local_feat_2 = vision(patches, local_feat_1)
                
                # Combine SAM and ViT features
                local_features = torch.cat([
                    local_feat_2[:, 1:],  # Skip CLS token
                    local_feat_1.flatten(2).permute(0, 2, 1)
                ], dim=-1)
                local_features = proj(local_features)
            
            # Global features
            global_feat_1 = sam(global_view)
            global_feat_2 = vision(global_view, global_feat_1)
            
            global_features = torch.cat([
                global_feat_2[:, 1:],
                global_feat_1.flatten(2).permute(0, 2, 1)
            ], dim=-1)
            global_features = proj(global_features)
            
            # Reshape and add newlines
            _, hw, dim = global_features.shape
            h = w = int(hw ** 0.5)
            
            global_features = global_features.view(h, w, dim)
            
            if newline is not None:
                global_features = torch.cat([
                    global_features,
                    newline[None, None, :].expand(h, 1, dim)
                ], dim=1)
            
            global_features = global_features.view(-1, dim)
            
            # Combine local and global
            if has_patches:
                width_crops, height_crops = crop_shape
                _, hw2, dim2 = local_features.shape
                h2 = w2 = int(hw2 ** 0.5)
                
                local_features = local_features.view(
                    height_crops, width_crops, h2, w2, dim2
                ).permute(0, 2, 1, 3, 4).reshape(
                    height_crops * h2, width_crops * w2, dim2
                )
                
                if newline is not None:
                    local_features = torch.cat([
                        local_features,
                        newline[None, None, :].expand(height_crops * h2, 1, dim2)
                    ], dim=1)
                
                local_features = local_features.view(-1, dim2)
                
                # Concatenate: local + global + separator
                features_list = [local_features, global_features]
            else:
                features_list = [global_features]
            
            if separator is not None:
                features_list.append(separator[None, :])
            
            combined = torch.cat(features_list, dim=0)
        
        return combined
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        images_seq_mask: Optional[torch.BoolTensor] = None,
        images_spatial_crop: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> DifferentiableOutput:
        """
        DeepSeek-specific forward pass with differentiable vision processing.
        
        This uses a hybrid approach:
        1. Process images through our differentiable vision pipeline
        2. Create embeddings with vision features scattered in
        3. Call the native model with inputs_embeds (bypassing its vision processing)
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            images: List of (patches, global_view) tuples
            images_seq_mask: Mask indicating image token positions [B, L]
            images_spatial_crop: Crop grid info [N, 2]
            labels: Training labels [B, L]
            **kwargs: Additional arguments
            
        Returns:
            DifferentiableOutput
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Get embeddings from input tokens
        embed_tokens = self.model.get_input_embeddings()
        embeddings = embed_tokens(input_ids)
        embeddings = embeddings.clone()  # Avoid in-place operations
        
        # Process and scatter image features
        if images is not None and images_seq_mask is not None:
            # Store images for gradient retrieval
            self._current_images = images
            
            for idx, (patches, global_view) in enumerate(images):
                # Skip if no image data
                if global_view.abs().sum().item() == 0:
                    continue
                
                # Get crop shape
                if images_spatial_crop is not None and idx < len(images_spatial_crop):
                    crop_shape = tuple(images_spatial_crop[idx].tolist())
                else:
                    crop_shape = (1, 1)
                
                # Process vision features using our differentiable method
                vision_features = self.process_vision_features(
                    patches, global_view, crop_shape
                )
                vision_features = vision_features.to(
                    device=embeddings.device, 
                    dtype=embeddings.dtype
                )
                
                # Scatter into embeddings at image token positions
                if idx < batch_size:
                    mask = images_seq_mask[idx].unsqueeze(-1)
                    # Make sure feature count matches mask
                    num_image_positions = mask.sum().item()
                    if vision_features.shape[0] >= num_image_positions:
                        vision_features_trimmed = vision_features[:int(num_image_positions)]
                    else:
                        # Pad if needed
                        pad_size = int(num_image_positions) - vision_features.shape[0]
                        vision_features_trimmed = torch.cat([
                            vision_features,
                            torch.zeros(pad_size, vision_features.shape[1], 
                                       device=vision_features.device, dtype=vision_features.dtype)
                        ], dim=0)
                    
                    embeddings[idx] = embeddings[idx].masked_scatter(
                        mask.expand_as(embeddings[idx]), 
                        vision_features_trimmed
                    )
        
        # The DeepSeek model requires input_ids and doesn't properly support inputs_embeds.
        # Solution: Patch the embedding layer to return our pre-computed embeddings.
        
        embed_layer = self.model.get_input_embeddings()
        original_forward = embed_layer.forward
        
        # Create a wrapper that returns our embeddings
        def patched_embed_forward(input_ids_inner):
            # Return our pre-computed embeddings instead of looking up from the table
            return embeddings
        
        # Create position_ids for rotary embeddings  
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=self.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Create zero images so the model skips its vision processing
        # The model checks: torch.sum(images[0][1]).item() != 0
        zero_global = torch.zeros(
            1, 3, self.ds_config["base_size"], self.ds_config["base_size"],
            dtype=self.dtype, device=self.device
        )
        zero_patches = torch.zeros(
            1, 3, self.ds_config["patch_size"], self.ds_config["patch_size"],
            dtype=self.dtype, device=self.device
        )
        dummy_images = [(zero_patches, zero_global)]
        dummy_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
        dummy_crop = torch.tensor([[1, 1]], dtype=torch.long, device=self.device)
        
        try:
            # Patch the embedding layer
            embed_layer.forward = patched_embed_forward
            
            with torch.enable_grad():
                # Call the model - it will use our patched embeddings
                # Zero images means it skips vision processing
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    images=dummy_images,
                    images_seq_mask=dummy_mask,
                    images_spatial_crop=dummy_crop,
                    use_cache=False,
                    return_dict=True,
                )
            
            logits = outputs.logits
            
        finally:
            # Always restore the original embedding forward
            embed_layer.forward = original_forward
        
        logits = logits.float()  # Ensure float for loss computation
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return DifferentiableOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
        )
    
    def forward_with_patched_vision(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        images_seq_mask: Optional[torch.BoolTensor] = None,
        images_spatial_crop: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> DifferentiableOutput:
        """
        Forward that patches the model's vision processing to enable gradients.
        
        This approach:
        1. Temporarily patches the vision encoder forward methods to use enable_grad
        2. Calls the model's native forward (which handles all LLM complexity)
        3. Restores original forwards
        
        The key insight: the native forward WORKS, it just has no_grad around
        vision processing. By patching the vision encoders, we enable gradients
        while keeping all the model's internal logic intact.
        """
        # Store images for gradient retrieval
        if images is not None:
            self._current_images = images
        
        model_base = self.model.model if hasattr(self.model, 'model') else self.model
        
        # Store original forwards
        original_sam_forward = None
        original_vision_forward = None
        sam_model = None
        vision_model = None
        
        # Find vision components
        if hasattr(model_base, 'sam_model') and model_base.sam_model is not None:
            sam_model = model_base.sam_model
            original_sam_forward = sam_model.forward
            
        if hasattr(model_base, 'vision_model') and model_base.vision_model is not None:
            vision_model = model_base.vision_model
            original_vision_forward = vision_model.forward
        
        # Create grad-enabled wrappers
        if original_sam_forward is not None:
            def patched_sam_forward(*args, **kwargs):
                with torch.enable_grad():
                    return original_sam_forward(*args, **kwargs)
            sam_model.forward = patched_sam_forward
            
        if original_vision_forward is not None:
            def patched_vision_forward(*args, **kwargs):
                with torch.enable_grad():
                    return original_vision_forward(*args, **kwargs)
            vision_model.forward = patched_vision_forward
        
        try:
            # Call the model's native forward with enable_grad context
            with torch.enable_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    images_seq_mask=images_seq_mask,
                    images_spatial_crop=images_spatial_crop,
                    labels=labels,
                    **kwargs,
                )
            
            return DifferentiableOutput(
                loss=getattr(outputs, 'loss', None),
                logits=getattr(outputs, 'logits', None),
                hidden_states=getattr(outputs, 'hidden_states', None),
            )
            
        finally:
            # Always restore original forwards
            if original_sam_forward is not None:
                sam_model.forward = original_sam_forward
            if original_vision_forward is not None:
                vision_model.forward = original_vision_forward
    
    def get_image_gradients(self) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get gradients for DeepSeek image format.
        
        Returns:
            List of (patches_grad, global_grad) tuples
        """
        if self._current_images is None:
            return None
        
        gradients = []
        for patches, global_view in self._current_images:
            patch_grad = patches.grad if patches.grad is not None else torch.zeros_like(patches)
            global_grad = global_view.grad if global_view.grad is not None else torch.zeros_like(global_view)
            gradients.append((patch_grad, global_grad))
        
        return gradients
    
    def verify_gradient_flow(self) -> bool:
        """
        DeepSeek-specific gradient flow verification.
        
        Creates test inputs in DeepSeek's expected format and verifies
        gradients flow back to the input images.
        
        Returns:
            True if gradients flow correctly
        """
        print("=" * 50)
        print("DEEPSEEK GRADIENT FLOW VERIFICATION")
        print("=" * 50)
        
        # Create test image with gradients (use randn, not zeros, to ensure model processes them)
        test_global = torch.randn(
            1, 3, self.ds_config["base_size"], self.ds_config["base_size"],
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        
        # Use randn for patches too - zeros might be skipped by the model
        test_patches = torch.randn(
            1, 3, self.ds_config["patch_size"], self.ds_config["patch_size"],
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        
        print(f"\n1. Test image shape: global={test_global.shape}, patches={test_patches.shape}")
        print(f"   requires_grad: global={test_global.requires_grad}, patches={test_patches.requires_grad}")
        
        print("\n2. Checking component gradients...")
        self.gradient_enabler.verify()
        
        print("\n3. Testing vision feature extraction...")
        vision_success = False
        try:
            with torch.enable_grad():
                features = self.process_vision_features(
                    test_patches, 
                    test_global,
                    crop_shape=(1, 1),
                )
            
            print(f"   ✓ Vision features shape: {features.shape}")
            
            # Backward through features (no retain_graph for compiled models)
            loss = features.sum()
            
            # Try to work around torch.compile donated buffer issue
            try:
                loss.backward()
            except RuntimeError as e:
                if "donated buffer" in str(e):
                    print("   ⚠ torch.compile donated buffer issue detected")
                    # Try disabling donated buffers
                    try:
                        from torch._functorch import config as functorch_config
                        functorch_config.donated_buffer = False
                        print("   ℹ Disabled donated_buffer, retrying...")
                        # Need fresh tensors after the failed backward
                        test_global_retry = torch.randn_like(test_global, requires_grad=True)
                        test_patches_retry = torch.zeros_like(test_patches, requires_grad=True)
                        features_retry = self.process_vision_features(
                            test_patches_retry, test_global_retry, crop_shape=(1, 1)
                        )
                        features_retry.sum().backward()
                        test_global = test_global_retry  # Use retried tensor for grad check
                    except Exception as e2:
                        print(f"   ⚠ Workaround failed: {e2}")
                        raise e
                else:
                    raise
            
            if test_global.grad is not None:
                grad_norm = test_global.grad.norm().item()
                print(f"   ✓ Global image gradient norm: {grad_norm:.6f}")
                vision_success = True
            else:
                print("   ✗ No gradient on global image")
                
        except Exception as e:
            print(f"   ✗ Vision feature test failed: {e}")
        
        # Reset for native forward test
        test_global2 = torch.randn(
            1, 3, self.ds_config["base_size"], self.ds_config["base_size"],
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        test_patches2 = torch.randn(
            1, 3, self.ds_config["patch_size"], self.ds_config["patch_size"],
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        
        print("\n4. Testing native model forward...")
        native_success = False
        
        # Calculate expected number of image tokens (needed for tests 4 and 5)
        num_queries_base = (self.ds_config["base_size"] // 16) // self.ds_config["downsample_ratio"]
        num_image_tokens = (num_queries_base + 1) * num_queries_base + 1
        
        try:
            # Create input_ids with image tokens
            seq_len = num_image_tokens + 10
            test_ids = torch.full(
                (1, seq_len),
                fill_value=1,  # Use token ID 1 as filler
                dtype=torch.long,
                device=self.device,
            )
            # Fill image token positions
            test_ids[0, :num_image_tokens] = self.image_token_id
            
            # Create images_seq_mask
            images_seq_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=self.device)
            images_seq_mask[0, :num_image_tokens] = True
            
            # Prepare images in DeepSeek format: list of (patches, global_view)
            images = [(test_patches2, test_global2)]
            images_spatial_crop = torch.tensor([[1, 1]], dtype=torch.long, device=self.device)
            
            # Call model's native forward with enable_grad
            with torch.enable_grad():
                output = self.model(
                    input_ids=test_ids,
                    images=images,
                    images_seq_mask=images_seq_mask,
                    images_spatial_crop=images_spatial_crop,
                )
            
            if hasattr(output, 'logits') and output.logits is not None:
                print(f"   ✓ Native forward succeeded, logits shape: {output.logits.shape}")
                
                # Backward
                loss = output.logits.sum()
                loss.backward()
                
                if test_global2.grad is not None:
                    grad_norm = test_global2.grad.norm().item()
                    print(f"   ✓ Native backward gradient norm: {grad_norm:.6f}")
                    native_success = True
                else:
                    print("   ✗ No gradient after native backward")
                    print("   Note: Model may use torch.no_grad() internally")
            else:
                print("   ✗ Native forward produced no logits")
                
        except Exception as e:
            print(f"   ✗ Native forward test failed: {e}")

        # Test 5: Custom forward method (inputs_embeds approach)
        test_global3 = torch.randn(
            1, 3, self.ds_config["base_size"], self.ds_config["base_size"],
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        test_patches3 = torch.randn(
            1, 3, self.ds_config["patch_size"], self.ds_config["patch_size"],
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        
        print("\n5. Testing pipeline.forward() (inputs_embeds approach)...")
        custom_success = False
        try:
            # Create inputs
            seq_len = num_image_tokens + 10
            test_ids = torch.full(
                (1, seq_len),
                fill_value=1,
                dtype=torch.long,
                device=self.device,
            )
            test_ids[0, :num_image_tokens] = self.image_token_id
            
            images_seq_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=self.device)
            images_seq_mask[0, :num_image_tokens] = True
            
            images = [(test_patches3, test_global3)]
            images_spatial_crop = torch.tensor([[1, 1]], dtype=torch.long, device=self.device)
            
            # Call our custom forward
            with torch.enable_grad():
                output = self.forward(
                    input_ids=test_ids,
                    images=images,
                    images_seq_mask=images_seq_mask,
                    images_spatial_crop=images_spatial_crop,
                )
            
            if output.logits is not None:
                print(f"   ✓ Custom forward succeeded, logits shape: {output.logits.shape}")
                
                loss = output.logits.sum()
                loss.backward()
                
                if test_global3.grad is not None:
                    grad_norm = test_global3.grad.norm().item()
                    print(f"   ✓ Custom backward gradient norm: {grad_norm:.6f}")
                    custom_success = True
                else:
                    print("   ✗ No gradient after custom backward")
            else:
                print("   ✗ Custom forward produced no logits")
                
        except Exception as e:
            print(f"   ✗ Custom forward test failed: {e}")
        
        # Test 6: Patched vision forward approach
        test_global4 = torch.randn(
            1, 3, self.ds_config["base_size"], self.ds_config["base_size"],
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        test_patches4 = torch.randn(
            1, 3, self.ds_config["patch_size"], self.ds_config["patch_size"],
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        
        print("\n6. Testing forward_with_patched_vision() (patched native approach)...")
        patched_success = False
        try:
            seq_len = num_image_tokens + 10
            test_ids = torch.full(
                (1, seq_len),
                fill_value=1,
                dtype=torch.long,
                device=self.device,
            )
            test_ids[0, :num_image_tokens] = self.image_token_id
            
            images_seq_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=self.device)
            images_seq_mask[0, :num_image_tokens] = True
            
            images = [(test_patches4, test_global4)]
            images_spatial_crop = torch.tensor([[1, 1]], dtype=torch.long, device=self.device)
            
            with torch.enable_grad():
                output = self.forward_with_patched_vision(
                    input_ids=test_ids,
                    images=images,
                    images_seq_mask=images_seq_mask,
                    images_spatial_crop=images_spatial_crop,
                )
            
            if output.logits is not None:
                print(f"   ✓ Patched forward succeeded, logits shape: {output.logits.shape}")
                
                loss = output.logits.sum()
                loss.backward()
                
                if test_global4.grad is not None:
                    grad_norm = test_global4.grad.norm().item()
                    print(f"   ✓ Patched backward gradient norm: {grad_norm:.6f}")
                    patched_success = True
                else:
                    print("   ✗ No gradient after patched backward")
            else:
                print("   ✗ Patched forward produced no logits")
                
        except Exception as e:
            print(f"   ✗ Patched forward test failed: {e}")
        
        # Summary
        print("\n" + "=" * 50)
        overall_success = vision_success and custom_success
        print(f"RESULT: {'✓ PASS' if overall_success else '✗ FAIL'}")
        print(f"  {'✓' if vision_success else '✗'} Vision features: {'gradients flow' if vision_success else 'gradients blocked'}")
        print(f"  {'✓' if native_success else '✗'} Native forward: {'gradients flow' if native_success else 'gradients blocked'}")
        print(f"  {'✓' if custom_success else '✗'} pipeline.forward(): {'gradients flow' if custom_success else 'gradients blocked'}")
        print(f"  {'✓' if patched_success else '✗'} forward_with_patched_vision(): {'gradients flow' if patched_success else 'gradients blocked'}")
        
        if overall_success:
            print("\n  ✓ Use pipeline.forward() for training!")
        elif vision_success:
            print("\n  ⚠ Vision encoder works but full forward failed.")
            print("  Consider using process_vision_features() directly in a custom training loop.")
        
        print("=" * 50)
        
        return overall_success
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        enable_image_gradients: bool = True,
        use_unsloth: bool = False,
        load_in_4bit: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> "DeepSeekOCRPipeline":
        """
        Load a DeepSeek model and create pipeline.
        
        Args:
            model_name_or_path: Model path or HF identifier
            enable_image_gradients: Enable gradient flow
            use_unsloth: Use Unsloth for loading
            load_in_4bit: 4-bit quantization
            verbose: Print status
            **kwargs: Additional loading args
            
        Returns:
            DeepSeekOCRPipeline instance
        """
        if verbose:
            print(f"Loading DeepSeek model: {model_name_or_path}")
        
        if use_unsloth:
            model, tokenizer = cls._load_with_unsloth(
                model_name_or_path,
                load_in_4bit=load_in_4bit,
                trust_remote_code=True,
                **kwargs,
            )
        else:
            model, tokenizer = cls._load_standard(
                model_name_or_path,
                trust_remote_code=True,
                **kwargs,
            )
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            enable_image_gradients=enable_image_gradients,
            verbose=verbose,
        )


class DeepSeekImageProcessor(DifferentiableImageProcessor):
    """
    Image processor specifically for DeepSeek models.
    
    Handles the specific preprocessing expected by DeepSeek:
    - Dynamic cropping for large images
    - SAM-compatible normalization
    - Proper patch sizing
    """
    
    def __init__(
        self,
        patch_size: int = 640,
        base_size: int = 1024,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        min_crops: int = 2,
        max_crops: int = 9,
    ):
        super().__init__(
            patch_size=patch_size,
            base_size=base_size,
            mean=mean,
            std=std,
            use_dynamic_crops=True,
            min_crops=min_crops,
            max_crops=max_crops,
        )
        
        # DeepSeek-specific settings
        self.vision_patch_size = 16
        self.downsample_ratio = 4
    
    def calculate_token_count(
        self,
        crop_grid: Tuple[int, int],
    ) -> int:
        """
        Calculate number of image tokens for a given crop configuration.
        
        Args:
            crop_grid: (width_crops, height_crops)
            
        Returns:
            Number of image tokens
        """
        num_queries = (self.patch_size // self.vision_patch_size) // self.downsample_ratio
        num_queries_base = (self.base_size // self.vision_patch_size) // self.downsample_ratio
        
        width_crops, height_crops = crop_grid
        
        # Global view tokens
        tokens = (num_queries_base + 1) * num_queries_base + 1
        
        # Local view tokens (if cropped)
        if width_crops > 1 or height_crops > 1:
            tokens += (num_queries * width_crops + 1) * (num_queries * height_crops)
        
        return tokens


class DeepSeekDataCollator:
    """
    Data collator for DeepSeek OCR training.
    
    Handles:
    - Message formatting
    - Image processing with gradients
    - Proper masking for training
    """
    
    def __init__(
        self,
        pipeline: DeepSeekOCRPipeline,
        max_length: int = 2048,
        train_on_response_only: bool = True,
    ):
        """
        Initialize collator.
        
        Args:
            pipeline: DeepSeekOCRPipeline instance
            max_length: Maximum sequence length
            train_on_response_only: Only compute loss on assistant responses
        """
        self.pipeline = pipeline
        self.tokenizer = pipeline.tokenizer
        self.image_processor = pipeline.image_processor
        self.max_length = max_length
        self.train_on_response_only = train_on_response_only
        self.image_token_id = pipeline.image_token_id
    
    def __call__(
        self,
        examples: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Expected example format:
        {
            "messages": [
                {"role": "user", "content": "Describe <image>", "images": [PIL.Image]},
                {"role": "assistant", "content": "This shows..."},
            ]
        }
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_images = []
        batch_seq_mask = []
        batch_spatial_crop = []
        
        device = self.pipeline.device
        dtype = self.pipeline.dtype
        
        for example in examples:
            messages = example.get("messages", [])
            
            # Process messages and images
            input_ids, images_seq_mask, images, spatial_crops = self._process_messages(
                messages, device, dtype
            )
            
            # Create labels
            labels = input_ids.clone()
            labels[images_seq_mask] = -100  # Don't predict image tokens
            
            if self.train_on_response_only:
                # Find assistant response start
                # This is simplified - may need conversation template matching
                labels[:len(labels)//2] = -100  # Rough approximation
            
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_seq_mask.append(images_seq_mask)
            batch_images.extend(images)
            batch_spatial_crop.extend(spatial_crops)
        
        # Pad sequences
        from torch.nn.utils.rnn import pad_sequence
        
        input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id or 0)
        labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)
        images_seq_mask = pad_sequence(batch_seq_mask, batch_first=True, padding_value=False)
        attention_mask = (input_ids != (self.tokenizer.pad_token_id or 0)).long()
        
        # Stack spatial crop info
        if batch_spatial_crop:
            images_spatial_crop = torch.stack([torch.tensor(c) for c in batch_spatial_crop])
        else:
            images_spatial_crop = torch.zeros(1, 2, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": batch_images,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }
    
    def _process_messages(
        self,
        messages: List[Dict],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
        """Process conversation messages."""
        all_tokens = []
        seq_mask = []
        images = []
        spatial_crops = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            msg_images = msg.get("images", [])
            
            # Add role prefix (simplified)
            role_prefix = f"<|{role.capitalize()}|>: "
            prefix_tokens = self.tokenizer.encode(role_prefix, add_special_tokens=False)
            all_tokens.extend(prefix_tokens)
            seq_mask.extend([False] * len(prefix_tokens))
            
            # Split content by image tags
            parts = content.split("<image>")
            img_idx = 0
            
            for i, part in enumerate(parts):
                # Add text
                if part:
                    text_tokens = self.tokenizer.encode(part, add_special_tokens=False)
                    all_tokens.extend(text_tokens)
                    seq_mask.extend([False] * len(text_tokens))
                
                # Add image placeholder
                if i < len(parts) - 1 and img_idx < len(msg_images):
                    pil_img = msg_images[img_idx]
                    
                    # Process image
                    image_tensor = pil_to_tensor(
                        pil_img, device=device, dtype=dtype,
                        requires_grad=self.pipeline.enable_image_gradients
                    )
                    patches, global_view, info = self.image_processor(
                        image_tensor, return_info=True
                    )
                    
                    images.append((patches, global_view.squeeze(0)))
                    spatial_crops.append(info["crop_grid"])
                    
                    # Calculate token count
                    token_count = self.image_processor.calculate_token_count(info["crop_grid"])
                    
                    all_tokens.extend([self.image_token_id] * token_count)
                    seq_mask.extend([True] * token_count)
                    
                    img_idx += 1
        
        # Add EOS
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id:
            all_tokens.append(self.tokenizer.eos_token_id)
            seq_mask.append(False)
        
        return (
            torch.tensor(all_tokens, dtype=torch.long),
            torch.tensor(seq_mask, dtype=torch.bool),
            images,
            spatial_crops,
        )


if __name__ == "__main__":
    print("DeepSeek OCR Pipeline Module")
    print()
    print("Usage:")
    print("  from deepseek_ocr_pipeline import DeepSeekOCRPipeline, DeepSeekDataCollator")
    print()
    print("  # Load model")
    print("  pipeline = DeepSeekOCRPipeline.from_pretrained(")
    print("      'deepseek-ai/deepseek-vl2-tiny',")
    print("      enable_image_gradients=True,")
    print("  )")
    print()
    print("  # Create collator")
    print("  collator = DeepSeekDataCollator(pipeline)")
    print()
    print("  # Train")
    print("  batch = collator(examples)")
    print("  output = pipeline.forward(**batch)")
    print("  output.loss.backward()")