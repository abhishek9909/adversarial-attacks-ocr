"""
DeepSeekOCR - Unified interface for DeepSeek Vision-Language Model

Handles:
- Model loading
- Inference (full generation until EOS)
- Single forward pass
- Gradient computation on images
- Pixel optimization

Usage:
    from deepseek_ocr import DeepSeekOCR
    
    model = DeepSeekOCR.load("./loaded/deepseek_ocr")
    response = model.infer(image, "What text is in this image?")
    print(response)
"""

import unsloth

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
from typing import Optional, List, Tuple, Union, Callable
from dataclasses import dataclass


@dataclass
class InferenceOutput:
    """Output from inference."""
    text: str
    tokens: List[int]
    

@dataclass  
class GradientOutput:
    """Output from gradient computation."""
    loss: float
    image_grad: torch.Tensor  # Gradient on the preprocessed image
    grad_norm: float
    
    # Backward compatibility aliases
    @property
    def global_view_grad(self):
        return self.image_grad
    
    @property
    def patches_grad(self):
        return torch.zeros_like(self.image_grad)


class DeepSeekOCR:
    """
    Unified interface for DeepSeek OCR model.
    
    Example:
        model = DeepSeekOCR.load("./loaded/deepseek_ocr")
        
        # Inference
        response = model.infer(image, "Describe this image")
        
        # Gradient computation
        grads = model.compute_gradients(image, target_text="Hello")
        
        # Pixel optimization
        optimized = model.optimize_pixels(image, target_text="Cat", steps=10)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        dtype: torch.dtype,
        image_token_id: int,
        config: dict,
        mode: str = "base",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.image_token_id = image_token_id
        self.config = config
        self.mode = mode  # Native resolution mode: tiny, small, base, large
        
        # Image normalization - must match DeepSeekImageProcessor (0.5, 0.5, 0.5)
        self.mean = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=dtype).view(1, 3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=dtype).view(1, 3, 1, 1)
    
    @classmethod
    def load(
        cls,
        model_path: str,
        enable_gradients: bool = True,
        mode: str = "base",
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> "DeepSeekOCR":
        """
        Load model from path.
        
        Args:
            model_path: Path to model directory
            enable_gradients: Enable gradient computation on images
            mode: Native resolution mode - "tiny" (512), "small" (640), 
                  "base" (1024), or "large" (1280)
            device: Device to use (auto-detected if None)
            verbose: Print loading info
            
        Returns:
            DeepSeekOCR instance
        """
        if verbose:
            print(f"Loading DeepSeekOCR from {model_path}...")
            print(f"Mode: {mode}")
        
        # Import here to avoid circular imports
        from deepseek_ocr_pipeline import DeepSeekOCRPipeline
        
        pipeline = DeepSeekOCRPipeline.from_pretrained(
            model_path,
            enable_image_gradients=enable_gradients,
            verbose=verbose,
            use_unsloth=True
        )
        
        config = {
            "base_size": pipeline.ds_config["base_size"],
            "patch_size": pipeline.ds_config["patch_size"],
            "downsample_ratio": pipeline.ds_config["downsample_ratio"],
        }
        
        instance = cls(
            model=pipeline.model,
            tokenizer=pipeline.tokenizer,
            device=pipeline.device,
            dtype=pipeline.dtype,
            image_token_id=pipeline.image_token_id,
            config=config,
            mode=mode,
        )
        
        # Store pipeline for access to helper methods
        instance._pipeline = pipeline
        
        if verbose:
            print(f"✓ Model loaded on {pipeline.device}")
        
        return instance
    
    # =========================================================================
    # Image Processing
    # =========================================================================
    
    def preprocess_image(
        self,
        image: Union[Image.Image, str, torch.Tensor],
        requires_grad: bool = False,
        mode: str = "base",
        use_pipeline_processor: bool = True,
    ) -> torch.Tensor:
        """
        Preprocess image for model input using native resolution mode.
        
        Native resolution modes (from DeepSeek-OCR paper):
        - tiny (512): Direct resize to 512x512 → 64 tokens
        - small (640): Direct resize to 640x640 → 100 tokens
        - base (1024): Resize maintaining aspect ratio, pad to 1024x1024 → 256 tokens
        - large (1280): Resize maintaining aspect ratio, pad to 1280x1280 → 400 tokens
        
        Args:
            image: PIL Image, path string, or tensor
            requires_grad: Enable gradients on output tensor
            mode: Resolution mode - "tiny", "small", "base", or "large"
            use_pipeline_processor: Use pipeline's image processor (recommended)
            
        Returns:
            Preprocessed image tensor [1, 3, H, W]
        """
        # Mode configurations
        mode_config = {
            "tiny": {"size": 512, "preserve_aspect": False},
            "small": {"size": 640, "preserve_aspect": False},
            "base": {"size": 1024, "preserve_aspect": True},
            "large": {"size": 1280, "preserve_aspect": True},
        }
        
        if mode not in mode_config:
            raise ValueError(f"Unknown mode: {mode}. Choose from {list(mode_config.keys())}")
        
        target_size = mode_config[mode]["size"]
        preserve_aspect = mode_config[mode]["preserve_aspect"]
        
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Convert PIL to processed tensor
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            
            if use_pipeline_processor and hasattr(self, '_pipeline'):
                # Use pipeline's differentiable processing for best compatibility
                from differentiable_transforms import pil_to_tensor, differentiable_resize, DifferentiableNormalize
                
                # Convert to tensor [0, 1]
                img_tensor = pil_to_tensor(
                    image, 
                    device=self.device, 
                    dtype=self.dtype,
                    requires_grad=requires_grad
                )
                
                # Resize to target size
                if preserve_aspect:
                    # Resize maintaining aspect ratio, then pad
                    img_tensor = self._differentiable_resize_and_pad(img_tensor, target_size)
                else:
                    # Direct resize
                    img_tensor = differentiable_resize(img_tensor, (target_size, target_size))
                
                # Normalize with (0.5, 0.5, 0.5)
                normalize = DifferentiableNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                normalize = normalize.to(self.device)
                img_tensor = normalize(img_tensor)
                
                # Add batch dim if needed
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                    
            else:
                # Fallback to PIL-based preprocessing
                if preserve_aspect:
                    processed = self._resize_and_pad(image, target_size)
                else:
                    processed = image.resize((target_size, target_size), Image.LANCZOS)
                
                to_tensor = T.ToTensor()
                img_tensor = to_tensor(processed).unsqueeze(0)
                img_tensor = img_tensor.to(device=self.device, dtype=self.dtype)
                img_tensor = (img_tensor - self.mean) / self.std
            
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            img_tensor = image.to(device=self.device, dtype=self.dtype)
        
        if requires_grad and not img_tensor.requires_grad:
            img_tensor = img_tensor.clone().requires_grad_(True)
        
        # For non-leaf tensors (after transforms), we need retain_grad() 
        # to access gradients after backward()
        if requires_grad and not img_tensor.is_leaf:
            img_tensor.retain_grad()
        
        return img_tensor
    
    def _differentiable_resize_and_pad(
        self, 
        image: torch.Tensor, 
        target_size: int,
    ) -> torch.Tensor:
        """Differentiable resize maintaining aspect ratio with padding."""
        from differentiable_transforms import differentiable_resize
        
        if image.dim() == 4:
            image = image.squeeze(0)
        
        C, H, W = image.shape
        
        # Calculate scale to fit within target while maintaining aspect ratio
        scale = min(target_size / H, target_size / W)
        new_h = int(H * scale)
        new_w = int(W * scale)
        
        # Resize
        resized = differentiable_resize(image, (new_h, new_w))
        
        # Pad to target size with 0.5 (which becomes 0 after normalization)
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # F.pad expects (left, right, top, bottom) for 2D
        padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.5)
        
        return padded
    
    def _resize_and_pad(self, image: Image.Image, target_size: int) -> Image.Image:
        """
        Resize image maintaining aspect ratio so max(h,w) = target_size,
        then pad to target_size x target_size.
        
        Padding uses gray (128, 128, 128) which becomes 0.5 in [0,1] space,
        and becomes 0 after normalization with mean=0.5, std=0.5.
        """
        w, h = image.size
        
        # Calculate scale to make max dimension = target_size
        scale = target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize maintaining aspect ratio
        resized = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Create padded image with gray padding (128 = 0.5 in 0-1 space)
        # After normalization (x - 0.5) / 0.5, gray becomes 0
        padded = Image.new("RGB", (target_size, target_size), (128, 128, 128))
        
        # Center the resized image
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        padded.paste(resized, (paste_x, paste_y))
        
        return padded
    
    def _prepare_model_images(
        self, 
        img_tensor: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare image tensor for model's expected format.
        
        For native resolution mode, we pass zeros for patches 
        so only the global view is processed.
        """
        # Create zero patches to skip local/SAM processing
        # The model expects (patches, global_view) tuples
        zero_patches = torch.zeros(
            1, 3, 640, 640,
            device=self.device,
            dtype=self.dtype,
        )
        
        # If img_tensor requires grad, we need patches to also support it
        # but since we want gradients only on global_view, keep patches as zeros
        if img_tensor.requires_grad:
            zero_patches = zero_patches.requires_grad_(True)
        
        return [(zero_patches, img_tensor)]
    
    def postprocess_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor back to PIL Image."""
        # Denormalize
        img = tensor * self.std + self.mean
        img = img.clamp(0, 1)
        
        # To numpy
        img = img[0].detach().cpu().float().permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        
        return Image.fromarray(img)
    
    # =========================================================================
    # Inference (Full Generation)
    # =========================================================================
    
    def infer_with_tensor(
        self,
        img_tensor: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        """
        Run inference with a pre-processed tensor (already normalized).
        
        Use this to test optimized tensors without postprocess/preprocess round-trip.
        
        Args:
            img_tensor: Preprocessed image tensor [1, 3, H, W] (already normalized)
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p sampling parameter
            do_sample: Use sampling vs greedy decoding
            
        Returns:
            Generated text response
        """
        # Ensure tensor is on correct device/dtype
        img_tensor = img_tensor.to(device=self.device, dtype=self.dtype)
        
        # Prepare for model
        images = self._prepare_model_images(img_tensor)
        
        # Prepare inputs
        inputs = self._prepare_inputs(prompt, images)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                images=images,
                images_seq_mask=inputs["images_seq_mask"],
                images_spatial_crop=inputs["images_spatial_crop"],
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0, input_len:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def infer(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        mode: Optional[str] = None,
    ) -> str:
        """
        Run inference - generate full response until EOS.
        
        Args:
            image: Input image (PIL Image or path)
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p sampling parameter
            do_sample: Use sampling vs greedy decoding
            mode: Override resolution mode (uses self.mode if None)
            
        Returns:
            Generated text response
        """
        mode = mode or self.mode
        
        # Preprocess image (single tensor for native resolution)
        img_tensor = self.preprocess_image(image, requires_grad=False, mode=mode)
        images = self._prepare_model_images(img_tensor)
        
        # Prepare inputs
        inputs = self._prepare_inputs(prompt, images)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                images=images,
                images_seq_mask=inputs["images_seq_mask"],
                images_spatial_crop=inputs["images_spatial_crop"],
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0, input_len:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def infer_streaming(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        max_new_tokens: int = 512,
        mode: Optional[str] = None,
    ):
        """
        Streaming inference - yields tokens as they're generated.
        
        Args:
            image: Input image
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            mode: Override resolution mode (uses self.mode if None)
            
        Yields:
            Generated text incrementally
        """
        mode = mode or self.mode
        
        img_tensor = self.preprocess_image(image, requires_grad=False, mode=mode)
        images = self._prepare_model_images(img_tensor)
        inputs = self._prepare_inputs(prompt, images)
        
        input_ids = inputs["input_ids"]
        generated_text = ""
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    images=images,
                    images_seq_mask=inputs["images_seq_mask"],
                    images_spatial_crop=inputs["images_spatial_crop"],
                )
                
                # Get next token (greedy)
                next_token_logits = outputs.logits[0, -1, :]
                next_token = next_token_logits.argmax().unsqueeze(0).unsqueeze(0)
                
                # Check EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Decode and yield
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                generated_text += token_text
                yield token_text
                
                # Append to input
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Update mask (extend with False for non-image tokens)
                new_mask = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
                inputs["images_seq_mask"] = torch.cat([inputs["images_seq_mask"], new_mask], dim=1)
    
    # =========================================================================
    # Forward Pass (Single Step)
    # =========================================================================
    
    def forward(
        self,
        image: Union[Image.Image, str, Tuple[torch.Tensor, torch.Tensor]],
        prompt: str,
        target_text: Optional[str] = None,
        requires_grad: bool = False,
        mode: Optional[str] = None,
    ):
        """
        Single forward pass through the model.
        
        Args:
            image: Input image or (patches, global_view) tuple
            prompt: Text prompt  
            target_text: Optional target for loss computation
            requires_grad: Enable gradients on image
            mode: Override resolution mode (uses self.mode if None)
            
        Returns:
            Model output with logits and optional loss
        """
        mode = mode or self.mode
        
        # Handle image input
        if isinstance(image, tuple):
            # Legacy format: (patches, global_view) - use global_view
            img_tensor = image[1]
        elif isinstance(image, torch.Tensor):
            img_tensor = image
        else:
            img_tensor = self.preprocess_image(image, requires_grad=requires_grad, mode=mode)
        
        images = self._prepare_model_images(img_tensor)
        
        # Prepare inputs
        inputs = self._prepare_inputs(prompt, images, target_text=target_text)
        
        # Forward through pipeline (supports gradients)
        output = self._pipeline.forward(
            input_ids=inputs["input_ids"],
            images=images,
            images_seq_mask=inputs["images_seq_mask"],
            images_spatial_crop=inputs["images_spatial_crop"],
            labels=inputs.get("labels"),
        )
        
        return output
    
    # =========================================================================
    # Gradient Computation
    # =========================================================================
    
    def compute_gradients(
        self,
        image: Union[Image.Image, str],
        target_text: str,
        prompt: str = "<image>\nDescribe this image:",
        loss_fn: Optional[Callable] = None,
        mode: Optional[str] = None,
    ) -> GradientOutput:
        """
        Compute gradients of loss with respect to image pixels.
        
        Model weights are NOT updated - only gradients are computed.
        
        Args:
            image: Input image
            target_text: Target output text
            prompt: Input prompt
            loss_fn: Optional custom loss function(logits, labels) -> loss
            mode: Override resolution mode (uses self.mode if None)
            
        Returns:
            GradientOutput with loss, gradients, and grad norm
        """
        mode = mode or self.mode
        
        # Preprocess with gradients
        img_tensor = self.preprocess_image(image, requires_grad=True, mode=mode)
        images = self._prepare_model_images(img_tensor)
        
        # Prepare inputs
        inputs = self._prepare_inputs(prompt, images, target_text=target_text)
        
        # Forward
        output = self._pipeline.forward(
            input_ids=inputs["input_ids"],
            images=images,
            images_seq_mask=inputs["images_seq_mask"],
            images_spatial_crop=inputs["images_spatial_crop"],
            labels=inputs["labels"],
        )
        
        # Compute loss
        if loss_fn is not None:
            loss = loss_fn(output.logits, inputs["labels"])
        else:
            loss = output.loss
        
        # Backward
        loss.backward()
        
        # Check if gradient exists
        if img_tensor.grad is None:
            raise RuntimeError(
                "Gradient not computed on img_tensor. "
                "This may happen if the tensor was detached somewhere in the pipeline. "
                f"is_leaf={img_tensor.is_leaf}, requires_grad={img_tensor.requires_grad}"
            )
        
        return GradientOutput(
            loss=loss.item(),
            image_grad=img_tensor.grad.clone(),
            grad_norm=img_tensor.grad.norm().item(),
        )
    
    def compute_gradients_at_original_resolution(
        self,
        image: Union[Image.Image, str],
        target_text: str,
        prompt: str = "<image>\nDescribe this image:",
        mode: Optional[str] = None,
    ) -> Tuple[float, torch.Tensor]:
        """
        Compute gradients at original image resolution.
        
        Uses differentiable resize so gradients flow back to original pixels.
        
        Args:
            image: Input image at any resolution
            target_text: Target output text
            prompt: Input prompt
            mode: Override resolution mode (uses self.mode if None)
            
        Returns:
            Tuple of (loss, gradient_at_original_resolution)
        """
        mode = mode or self.mode
        
        # Mode configurations
        mode_sizes = {"tiny": 512, "small": 640, "base": 1024, "large": 1280}
        target_size = mode_sizes.get(mode, 1024)
        
        # Load image without resizing
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Convert to tensor at original size
        original = T.ToTensor()(image).unsqueeze(0).to(
            device=self.device, dtype=self.dtype
        ).requires_grad_(True)
        
        # Differentiable resize to model input size
        img_tensor = F.interpolate(original, size=(target_size, target_size), mode='bilinear', align_corners=False)
        
        # Normalize
        img_tensor = (img_tensor - self.mean) / self.std
        
        # Prepare for model (zeros for patches)
        images = self._prepare_model_images(img_tensor)
        
        # Prepare inputs
        inputs = self._prepare_inputs(prompt, images, target_text=target_text)
        
        # Forward
        output = self._pipeline.forward(
            input_ids=inputs["input_ids"],
            images=images,
            images_seq_mask=inputs["images_seq_mask"],
            images_spatial_crop=inputs["images_spatial_crop"],
            labels=inputs["labels"],
        )
        
        loss = output.loss
        loss.backward()
        
        # Check if gradient exists
        if original.grad is None:
            raise RuntimeError(
                "Gradient not computed on original image tensor. "
                f"is_leaf={original.is_leaf}, requires_grad={original.requires_grad}"
            )
        
        # Gradient is at original resolution!
        return loss.item(), original.grad.clone()
    
    # =========================================================================
    # Pixel Optimization
    # =========================================================================
    
    def detect_region_by_color(
        self,
        image: Union[Image.Image, str],
        color: Tuple[int, int, int] = (255, 0, 0),  # Red by default
        tolerance: int = 30,
        return_mask: bool = False,
    ) -> Union[Tuple[int, int, int, int], Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        Detect a region marked by a specific color (e.g., painted border).
        
        Args:
            image: Input image
            color: RGB color to detect (default: red)
            tolerance: Color matching tolerance (0-255)
            return_mask: If True, also return the binary mask
            
        Returns:
            (x1, y1, x2, y2) bounding box in original image coordinates
            Or ((x1, y1, x2, y2), mask) if return_mask=True
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        img_np = np.array(image)
        
        # Create color mask with tolerance
        color = np.array(color)
        diff = np.abs(img_np.astype(np.int32) - color.astype(np.int32))
        mask = np.all(diff <= tolerance, axis=-1).astype(np.uint8)
        
        # Find bounding box of the colored region
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            raise ValueError(f"No pixels found matching color {color} with tolerance {tolerance}")
        
        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()
        
        # Add small padding
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img_np.shape[1], x2 + pad)
        y2 = min(img_np.shape[0], y2 + pad)
        
        if return_mask:
            return (x1, y1, x2, y2), mask
        return (x1, y1, x2, y2)
    
    def detect_region_by_contour(
        self,
        image: Union[Image.Image, str],
        color: Tuple[int, int, int] = (255, 0, 0),
        tolerance: int = 30,
    ) -> Tuple[int, int, int, int]:
        """
        Detect the INTERIOR region bounded by a colored border/contour.
        
        Useful when you draw a border around an area you want to optimize.
        
        Args:
            image: Input image
            color: Border color to detect (default: red)
            tolerance: Color matching tolerance
            
        Returns:
            (x1, y1, x2, y2) bounding box of the interior region
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        img_np = np.array(image)
        
        # Detect the colored border
        color = np.array(color)
        diff = np.abs(img_np.astype(np.int32) - color.astype(np.int32))
        border_mask = np.all(diff <= tolerance, axis=-1).astype(np.uint8) * 255
        
        try:
            import cv2
            # Find contours
            contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                raise ValueError("No contours found")
            
            # Get largest contour (assumes the border is the largest colored region)
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            return (x, y, x + w, y + h)
            
        except ImportError:
            # Fallback without OpenCV - just use the bbox of colored pixels
            print("Warning: OpenCV not available, using simple bbox detection")
            return self.detect_region_by_color(image, color, tolerance)
    
    def segment_with_sam(
        self,
        image: Union[Image.Image, str],
        point: Optional[Tuple[int, int]] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
        """
        Segment a region using SAM (Segment Anything Model).
        
        Requires: pip install segment-anything
        
        Args:
            image: Input image
            point: (x, y) point prompt inside the region to segment
            box: (x1, y1, x2, y2) box prompt around the region
            
        Returns:
            ((x1, y1, x2, y2), mask) - bounding box and binary mask
        """
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "SAM not installed. Run: pip install segment-anything\n"
                "And download weights from https://github.com/facebookresearch/segment-anything"
            )
        
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        img_np = np.array(image)
        
        # Initialize SAM (user needs to have weights downloaded)
        # This is a simplified version - in practice you'd cache the predictor
        sam_checkpoint = "sam_vit_b_01ec64.pth"  # User needs to provide this
        model_type = "vit_b"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(self.device)
        predictor = SamPredictor(sam)
        predictor.set_image(img_np)
        
        # Generate mask
        if point is not None:
            point_coords = np.array([[point[0], point[1]]])
            point_labels = np.array([1])  # 1 = foreground
            masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
        elif box is not None:
            box_np = np.array([box[0], box[1], box[2], box[3]])
            masks, _, _ = predictor.predict(
                box=box_np,
                multimask_output=False,
            )
        else:
            raise ValueError("Provide either point or box prompt")
        
        mask = masks[0]
        
        # Get bounding box from mask
        coords = np.where(mask > 0)
        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()
        
        return (x1, y1, x2, y2), mask
    
    def transform_bbox_to_model_space(
        self,
        bbox: Tuple[int, int, int, int],
        original_size: Tuple[int, int],
        mode: Optional[str] = None,
    ) -> Tuple[int, int, int, int]:
        """
        Transform bbox from original image coordinates to preprocessed tensor coordinates.
        
        Args:
            bbox: (x1, y1, x2, y2) in original image space
            original_size: (width, height) of original image
            mode: Resolution mode (uses self.mode if None)
            
        Returns:
            (x1, y1, x2, y2) in model tensor space
        """
        mode = mode or self.mode
        mode_config = {
            "tiny": {"size": 512, "preserve_aspect": False},
            "small": {"size": 640, "preserve_aspect": False},
            "base": {"size": 1024, "preserve_aspect": True},
            "large": {"size": 1280, "preserve_aspect": True},
        }
        
        target_size = mode_config[mode]["size"]
        preserve_aspect = mode_config[mode]["preserve_aspect"]
        
        orig_w, orig_h = original_size
        x1, y1, x2, y2 = bbox
        
        if preserve_aspect:
            # Same logic as _resize_and_pad
            scale = target_size / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            
            # Padding offsets (centered)
            pad_x = (target_size - new_w) // 2
            pad_y = (target_size - new_h) // 2
            
            # Transform coordinates
            x1_new = int(x1 * scale) + pad_x
            y1_new = int(y1 * scale) + pad_y
            x2_new = int(x2 * scale) + pad_x
            y2_new = int(y2 * scale) + pad_y
        else:
            # Direct resize (squish)
            scale_x = target_size / orig_w
            scale_y = target_size / orig_h
            
            x1_new = int(x1 * scale_x)
            y1_new = int(y1 * scale_y)
            x2_new = int(x2 * scale_x)
            y2_new = int(y2 * scale_y)
        
        # Clamp to valid range
        x1_new = max(0, min(x1_new, target_size))
        y1_new = max(0, min(y1_new, target_size))
        x2_new = max(0, min(x2_new, target_size))
        y2_new = max(0, min(y2_new, target_size))
        
        return (x1_new, y1_new, x2_new, y2_new)
    
    def transform_mask_to_model_space(
        self,
        mask: np.ndarray,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Transform a mask from original image space to model tensor space.
        
        Args:
            mask: Binary mask [H, W] in original image space
            mode: Resolution mode (uses self.mode if None)
            
        Returns:
            Mask tensor [1, 1, H, W] in model space
        """
        mode = mode or self.mode
        mode_config = {
            "tiny": {"size": 512, "preserve_aspect": False},
            "small": {"size": 640, "preserve_aspect": False},
            "base": {"size": 1024, "preserve_aspect": True},
            "large": {"size": 1280, "preserve_aspect": True},
        }
        
        target_size = mode_config[mode]["size"]
        preserve_aspect = mode_config[mode]["preserve_aspect"]
        
        orig_h, orig_w = mask.shape[:2]
        
        # Convert to tensor
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        
        if preserve_aspect:
            # Resize maintaining aspect ratio
            scale = target_size / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            
            mask_resized = F.interpolate(mask_tensor, size=(new_h, new_w), mode='nearest')
            
            # Pad to target size
            pad_x = (target_size - new_w) // 2
            pad_y = (target_size - new_h) // 2
            pad_x2 = target_size - new_w - pad_x
            pad_y2 = target_size - new_h - pad_y
            
            mask_padded = F.pad(mask_resized, (pad_x, pad_x2, pad_y, pad_y2), value=0)
        else:
            # Direct resize
            mask_padded = F.interpolate(mask_tensor, size=(target_size, target_size), mode='nearest')
        
        return mask_padded.to(device=self.device, dtype=self.dtype)

    def save_tensor_for_reload(
        self,
        tensor: torch.Tensor,
        path: str,
        format: str = "pt",
    ) -> str:
        """
        Save optimized tensor in a format that can be reloaded exactly.
        
        Args:
            tensor: Optimized tensor [1, 3, H, W] (already normalized)
            path: Base path (extension added based on format)
            format: "pt" (torch tensor), "npz" (numpy), or "png16" (16-bit PNG, lossy)
            
        Returns:
            Actual saved path
        """
        tensor = tensor.detach().cpu()
        
        if format == "pt":
            # Best option - exact preservation
            save_path = path if path.endswith('.pt') else f"{path}.pt"
            torch.save({
                'tensor': tensor,
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5],
                'normalized': True,
            }, save_path)
            
        elif format == "npz":
            # Numpy format - exact preservation
            save_path = path if path.endswith('.npz') else f"{path}.npz"
            np.savez(
                save_path,
                tensor=tensor.float().numpy(),
                mean=np.array([0.5, 0.5, 0.5]),
                std=np.array([0.5, 0.5, 0.5]),
                normalized=True,
            )
            
        elif format == "png16":
            # 16-bit PNG per channel - some precision loss but viewable
            # Denormalize first
            denorm = tensor * 0.5 + 0.5  # [0, 1] range
            denorm = denorm.clamp(0, 1)
            
            # Convert to 16-bit per channel
            img_np = (denorm[0].permute(1, 2, 0).float().numpy() * 65535).astype(np.uint16)
            
            # Save as 16-bit TIFF (PNG doesn't support 16-bit RGB well)
            save_path = path if path.endswith('.tiff') else f"{path}.tiff"
            from PIL import Image
            # PIL doesn't handle 16-bit RGB well, use imageio if available
            try:
                import imageio
                imageio.imwrite(save_path, img_np)
            except ImportError:
                # Fallback: save as separate 16-bit grayscale channels
                save_path = path if path.endswith('.npz') else f"{path}.npz"
                np.savez(save_path, tensor=tensor.float().numpy(), normalized=True)
                print("Warning: imageio not available, saved as npz instead")
        else:
            raise ValueError(f"Unknown format: {format}. Use 'pt', 'npz', or 'png16'")
        
        print(f"Saved tensor to {save_path}")
        return save_path
    
    def load_tensor_for_inference(
        self,
        path: str,
    ) -> torch.Tensor:
        """
        Load a saved tensor for direct inference.
        
        Args:
            path: Path to saved tensor (.pt or .npz)
            
        Returns:
            Tensor ready for infer_with_tensor()
        """
        if path.endswith('.pt'):
            data = torch.load(path, map_location=self.device)
            tensor = data['tensor']
        elif path.endswith('.npz'):
            data = np.load(path)
            tensor = torch.from_numpy(data['tensor'])
        else:
            raise ValueError(f"Unknown format: {path}. Use .pt or .npz files")
        
        return tensor.to(device=self.device, dtype=self.dtype)
    
    def optimize_pixels(
        self,
        image: Union[Image.Image, str],
        target_text: str,
        prompt: str = "<image>\nDescribe this image:",
        num_steps: int = 10,
        lr: float = 0.01,
        use_sign: bool = True,
        epsilon: Optional[float] = None,
        verbose: bool = True,
        mode: Optional[str] = None,
        return_tensor: bool = False,
        add_eos: bool = False,
        mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        original_coords: bool = True,
    ) -> Union[Image.Image, Tuple[Image.Image, torch.Tensor]]:
        """
        Optimize image pixels to produce target text.
        
        Model weights stay frozen - only pixels are updated.
        
        Args:
            image: Input image
            target_text: Target output text
            prompt: Input prompt
            num_steps: Number of optimization steps
            lr: Learning rate for pixel updates
            use_sign: Use sign of gradient (FGSM-style)
            epsilon: Max perturbation (None = unlimited)
            verbose: Print progress
            mode: Override resolution mode (uses self.mode if None)
            return_tensor: If True, also return the raw optimized tensor
            add_eos: If True, also learn to output EOS token after target_text
            mask: Optional mask. If original_coords=True, should be in original image space
            bbox: Optional (x1, y1, x2, y2). If original_coords=True, in original image space
            original_coords: If True, bbox/mask are in original image coordinates and will 
                           be transformed to model space automatically (default: True)
            
        Returns:
            If return_tensor=False: Optimized PIL image
            If return_tensor=True: Tuple of (PIL image, raw tensor for inference)
        """
        mode = mode or self.mode
        
        # Load image to get original size for coordinate transformation
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB") if isinstance(image, Image.Image) else None
        
        original_size = pil_image.size if pil_image else None  # (width, height)
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Preprocess (single tensor for native resolution)
        img_tensor = self.preprocess_image(pil_image or image, requires_grad=True, mode=mode)
        
        # Store original for masked optimization
        original_tensor = img_tensor.detach().clone()
        
        # Transform coordinates if needed
        if original_coords and original_size is not None:
            if bbox is not None:
                bbox = self.transform_bbox_to_model_space(bbox, original_size, mode)
                if verbose:
                    print(f"Transformed bbox to model space: {bbox}")
            if mask is not None:
                if isinstance(mask, np.ndarray):
                    mask = self.transform_mask_to_model_space(mask, mode)
                elif isinstance(mask, torch.Tensor):
                    # Assume it's already a numpy-like mask, convert and transform
                    mask_np = mask.cpu().numpy() if mask.dim() <= 2 else mask.squeeze().cpu().numpy()
                    mask = self.transform_mask_to_model_space(mask_np, mode)
        
        # Create optimization mask
        opt_mask = self._create_optimization_mask(img_tensor, mask, bbox)
        
        # Ensure we can get gradients (might be non-leaf after transforms)
        if not img_tensor.is_leaf:
            img_tensor.retain_grad()
        
        # Get inputs with optional EOS
        images = self._prepare_model_images(img_tensor)
        inputs = self._prepare_inputs(prompt, images, target_text=target_text, add_eos=add_eos)
        
        if verbose:
            print(f"Optimizing for: '{target_text}'" + (" + EOS" if add_eos else ""))
            print(f"Steps: {num_steps}, LR: {lr}")
            if opt_mask is not None:
                mask_ratio = opt_mask.sum() / opt_mask.numel()
                print(f"Optimizing {mask_ratio*100:.1f}% of pixels")
        
        for step in range(num_steps):
            # Zero gradients
            if img_tensor.grad is not None:
                img_tensor.grad.zero_()
            
            # Forward (rebuild images with current tensor)
            images = self._prepare_model_images(img_tensor)
            output = self._pipeline.forward(
                input_ids=inputs["input_ids"],
                images=images,
                images_seq_mask=inputs["images_seq_mask"],
                images_spatial_crop=inputs["images_spatial_crop"],
                labels=inputs["labels"],
            )
            
            loss = output.loss
            loss.backward()
            
            if verbose:
                print(f"  Step {step+1}/{num_steps}: loss={loss.item():.4f}")
            
            # Check gradient exists
            if img_tensor.grad is None:
                raise RuntimeError(f"No gradient on img_tensor at step {step+1}")
            
            # Update pixels (with optional masking)
            with torch.no_grad():
                grad = img_tensor.grad
                
                # Apply mask to gradient (zero out gradients outside mask)
                if opt_mask is not None:
                    grad = grad * opt_mask
                
                if use_sign:
                    update = lr * grad.sign()
                else:
                    update = lr * grad
                
                img_tensor = img_tensor - update
                
                if epsilon is not None:
                    # Clamp perturbation from original
                    perturbation = img_tensor - original_tensor
                    perturbation = torch.clamp(perturbation, -epsilon, epsilon)
                    img_tensor = original_tensor + perturbation
                
                # For masked optimization, ensure unmasked regions stay unchanged
                if opt_mask is not None:
                    img_tensor = torch.where(opt_mask > 0, img_tensor, original_tensor)
            
            img_tensor = img_tensor.detach().requires_grad_(True)
            
            # For non-leaf tensors, need retain_grad again
            if not img_tensor.is_leaf:
                img_tensor.retain_grad()
        
        if verbose:
            print(f"✓ Done. Final loss: {loss.item():.4f}")
        
        # Return both postprocessed image and raw tensor if requested
        if return_tensor:
            # Clone tensor for inference use (already normalized, ready for model)
            inference_tensor = img_tensor.detach().clone()
            pil_image = self.postprocess_image(img_tensor)
            return pil_image, inference_tensor
        else:
            return self.postprocess_image(img_tensor)
    
    def _create_optimization_mask(
        self,
        img_tensor: torch.Tensor,
        mask: Optional[Union[torch.Tensor, np.ndarray]],
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> Optional[torch.Tensor]:
        """Create mask for regional optimization."""
        if mask is not None and bbox is not None:
            raise ValueError("Provide either mask or bbox, not both")
        
        if mask is not None:
            # Convert numpy to tensor if needed
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask.astype(np.float32))
            
            # Ensure correct shape [1, 1, H, W]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(0)
            
            # Resize mask to match image tensor if needed
            if mask.shape[2:] != img_tensor.shape[2:]:
                mask = F.interpolate(
                    mask.float(), 
                    size=img_tensor.shape[2:], 
                    mode='nearest'
                )
            
            return mask.to(device=self.device, dtype=self.dtype)
        
        elif bbox is not None:
            x1, y1, x2, y2 = bbox
            _, _, H, W = img_tensor.shape
            
            # Clamp bbox to image bounds
            x1, x2 = max(0, x1), min(W, x2)
            y1, y2 = max(0, y1), min(H, y2)
            
            # Create binary mask
            mask = torch.zeros(1, 1, H, W, device=self.device, dtype=self.dtype)
            mask[:, :, y1:y2, x1:x2] = 1.0
            
            return mask
        
        return None
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _prepare_inputs(
        self,
        prompt: str,
        images: List[Tuple[torch.Tensor, torch.Tensor]],
        target_text: Optional[str] = None,
        add_eos: bool = False,
    ) -> dict:
        """Prepare model inputs from prompt and images."""
        # Get the image tensor (second element: global_view)
        patches, img_tensor = images[0]
        
        # Get exact token count by running through vision pipeline
        with torch.no_grad():
            features = self._pipeline.process_vision_features(
                patches.detach(), img_tensor.detach(), (1, 1)
            )
            num_image_tokens = features.shape[0]
        
        # Tokenize prompt
        # Ensure <image> is in prompt
        if "<image>" not in prompt:
            prompt = "<image>\n" + prompt
            
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        # Replace <image> token with image_token_id repeated
        input_ids = []
        image_token = self.tokenizer.convert_tokens_to_ids("<image>")
        
        for tid in prompt_tokens:
            if tid == image_token:
                input_ids.extend([self.image_token_id] * num_image_tokens)
            else:
                input_ids.append(tid)
        
        # Add target text if provided
        labels = None
        if target_text:
            target_tokens = self.tokenizer.encode(target_text, add_special_tokens=False)
            
            # Optionally add EOS token to learn sequence termination
            if add_eos and self.tokenizer.eos_token_id is not None:
                target_tokens = target_tokens + [self.tokenizer.eos_token_id]
            
            # Create labels (mask prompt with -100, keep target)
            labels = [-100] * len(input_ids) + target_tokens
            input_ids = input_ids + target_tokens
            
            labels = torch.tensor([labels], dtype=torch.long, device=self.device)
        
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        seq_len = input_ids.shape[1]
        
        # Create image sequence mask
        images_seq_mask = (input_ids == self.image_token_id)
        
        # Spatial crop info
        images_spatial_crop = torch.tensor([[1, 1]], dtype=torch.long, device=self.device)
        
        # Attention mask
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
            "labels": labels,
            "num_image_tokens": num_image_tokens,
        }
    
    def set_mode(self, mode: str):
        """Set the resolution mode."""
        valid_modes = ["tiny", "small", "base", "large"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Choose from {valid_modes}")
        self.mode = mode
        print(f"Mode set to: {mode}")
    
    def __repr__(self):
        return f"DeepSeekOCR(mode={self.mode}, device={self.device}, dtype={self.dtype})"