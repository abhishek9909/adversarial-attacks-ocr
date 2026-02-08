"""
Differentiable Image Resizing

Gradients flow through the resize operation back to original pixels.

Original image (e.g., 512x512)
    ↓ F.interpolate (differentiable)
Resized image (1024x1024)
    ↓ Model forward
Loss
    ↓ backward()
Gradients flow back through interpolate to original!
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np


def differentiable_resize(
    tensor: torch.Tensor,
    size: tuple,
    mode: str = 'bilinear',
) -> torch.Tensor:
    """
    Resize tensor in a differentiable way.
    
    Args:
        tensor: [B, C, H, W] image tensor with requires_grad=True
        size: (height, width) target size
        mode: 'bilinear', 'bicubic', 'nearest', or 'area'
        
    Returns:
        Resized tensor (gradients will flow back to input)
    """
    return F.interpolate(
        tensor,
        size=size,
        mode=mode,
        align_corners=False if mode in ['bilinear', 'bicubic'] else None,
    )


def demo_gradient_through_resize():
    """Demonstrate that gradients flow through resize."""
    print("=" * 50)
    print("DEMO: Gradients through resize")
    print("=" * 50)
    
    # Original small image
    original = torch.randn(1, 3, 64, 64, requires_grad=True)
    print(f"Original: {original.shape}, requires_grad={original.requires_grad}")
    
    # Resize to larger
    resized = differentiable_resize(original, (256, 256))
    print(f"Resized:  {resized.shape}, requires_grad={resized.requires_grad}")
    
    # Some "loss" on the resized image
    loss = resized.mean()
    loss.backward()
    
    # Check gradient on ORIGINAL
    print(f"\n✓ Original has gradient: {original.grad is not None}")
    print(f"✓ Gradient shape: {original.grad.shape}")
    print(f"✓ Gradient norm: {original.grad.norm().item():.6f}")


class DifferentiableImageProcessor:
    """
    Process images with differentiable resize.
    Gradients flow back to original resolution.
    """
    
    def __init__(
        self,
        pipeline,
        model_input_size: tuple = (1024, 1024),
        patch_size: tuple = (640, 640),
        resize_mode: str = 'bilinear',
    ):
        self.pipeline = pipeline
        self.model_input_size = model_input_size
        self.patch_size = patch_size
        self.resize_mode = resize_mode
        
        # Freeze model
        for param in pipeline.model.parameters():
            param.requires_grad = False
    
    def pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL to tensor (no resize yet)."""
        image = image.convert("RGB")
        tensor = T.ToTensor()(image)  # [C, H, W], values in [0, 1]
        return tensor.unsqueeze(0).to(
            device=self.pipeline.device,
            dtype=self.pipeline.dtype,
        )
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor back to PIL."""
        img = tensor[0].detach().cpu().float().clamp(0, 1)  # .float() converts bf16 to fp32
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(img)
    
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization (differentiable)."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device, dtype=tensor.dtype)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device, dtype=tensor.dtype)
        return (tensor - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Remove ImageNet normalization."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device, dtype=tensor.dtype)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device, dtype=tensor.dtype)
        return tensor * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    
    def prepare_for_model(
        self,
        original_tensor: torch.Tensor,
    ) -> tuple:
        """
        Prepare image for model with differentiable resize.
        
        Args:
            original_tensor: [1, 3, H, W] at original resolution, requires_grad=True
            
        Returns:
            (global_view, patches) - both connected to original via autograd
        """
        # Differentiable resize to model input size
        global_view = differentiable_resize(
            original_tensor,
            self.model_input_size,
            mode=self.resize_mode,
        )
        
        # Normalize (also differentiable)
        global_view = self.normalize(global_view)
        
        # Create patches (resize from original too for consistency)
        patches = differentiable_resize(
            original_tensor,
            self.patch_size,
            mode=self.resize_mode,
        )
        patches = self.normalize(patches)
        
        return global_view, patches
    
    def compute_loss_and_gradients(
        self,
        original_tensor: torch.Tensor,
        target_text: str,
        prompt: str = "<image>\nDescribe:",
    ) -> tuple:
        """
        Compute loss and gradients on ORIGINAL resolution image.
        
        Args:
            original_tensor: [1, 3, H, W] original image with requires_grad=True
            target_text: Target output text
            prompt: Input prompt
            
        Returns:
            (loss, gradient_on_original)
        """
        # Zero existing gradients
        if original_tensor.grad is not None:
            original_tensor.grad.zero_()
        
        # Differentiable resize
        global_view, patches = self.prepare_for_model(original_tensor)
        
        # Prepare inputs
        inputs = self._prepare_inputs(global_view, patches, prompt, target_text)
        
        # Forward
        output = self.pipeline.forward(
            input_ids=inputs['input_ids'],
            images=[(patches, global_view)],
            images_seq_mask=inputs['images_seq_mask'],
            images_spatial_crop=inputs['images_spatial_crop'],
            labels=inputs['labels'],
        )
        
        loss = output.loss
        
        # Backward - gradients flow through resize to original!
        loss.backward()
        
        return loss.item(), original_tensor.grad.clone()
    
    def _prepare_inputs(self, global_view, patches, prompt, target_text):
        """Prepare model inputs."""
        # Get number of image tokens
        with torch.no_grad():
            features = self.pipeline.process_vision_features(
                patches.detach(), global_view.detach(), (1, 1)
            )
            num_img_tokens = features.shape[0]
        
        tokenizer = self.pipeline.tokenizer
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
        
        # Build input_ids
        input_ids = []
        for tid in prompt_tokens:
            if tid == tokenizer.convert_tokens_to_ids("<image>"):
                input_ids.extend([self.pipeline.image_token_id] * num_img_tokens)
            else:
                input_ids.append(tid)
        
        input_ids = input_ids + target_tokens
        input_ids = torch.tensor([input_ids], device=self.pipeline.device)
        seq_len = input_ids.shape[1]
        
        labels = input_ids.clone()
        labels[0, :-len(target_tokens)] = -100
        
        images_seq_mask = (input_ids == self.pipeline.image_token_id)
        images_spatial_crop = torch.tensor([[1, 1]], device=self.pipeline.device)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'images_seq_mask': images_seq_mask,
            'images_spatial_crop': images_spatial_crop,
        }
    
    def optimize_original_pixels(
        self,
        image: Image.Image,
        target_text: str,
        num_steps: int = 10,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> Image.Image:
        """
        Optimize pixels at ORIGINAL resolution.
        
        The image is resized for the model, but gradients flow back
        to update the original resolution pixels.
        """
        # Convert to tensor at ORIGINAL resolution
        original = self.pil_to_tensor(image)
        original.requires_grad_(True)
        
        original_size = original.shape[2:]  # (H, W)
        
        if verbose:
            print(f"Original size: {original_size}")
            print(f"Model input size: {self.model_input_size}")
            print(f"Optimizing for: '{target_text}'")
        
        for step in range(num_steps):
            loss, grad = self.compute_loss_and_gradients(
                original, target_text
            )
            
            if verbose:
                print(f"  Step {step+1}/{num_steps}: loss={loss:.4f}, grad_norm={grad.norm().item():.4f}")
            
            # Update original pixels
            with torch.no_grad():
                original = original - lr * grad.sign()
                original = original.clamp(0, 1)  # Keep valid pixel range
            
            original = original.detach().requires_grad_(True)
        
        return self.tensor_to_pil(original)


def optimize_with_original_resolution(
    pipeline,
    image: Image.Image,
    target_text: str,
    num_steps: int = 10,
    lr: float = 0.01,
):
    """
    Simple function to optimize image at original resolution.
    
    Gradients flow: loss → resized_image → original_image
    """
    print("=" * 50)
    print("OPTIMIZING AT ORIGINAL RESOLUTION")
    print("=" * 50)
    
    processor = DifferentiableImageProcessor(pipeline)
    
    result = processor.optimize_original_pixels(
        image=image,
        target_text=target_text,
        num_steps=num_steps,
        lr=lr,
        verbose=True,
    )
    
    print("✓ Done! Gradients flowed through resize to original pixels.")
    return result


# if __name__ == "__main__":
#     # First, demo that gradients flow through resize
#     demo_gradient_through_resize()
    
#     print("\n")
    
#     # Then test with actual model
#     import sys
#     from deepseek_ocr_pipeline import DeepSeekOCRPipeline
    
#     model_path = sys.argv[1] if len(sys.argv) > 1 else "./loaded/deepseek_ocr"
    
#     print("Loading model...")
#     pipeline = DeepSeekOCRPipeline.from_pretrained(
#         model_path,
#         enable_image_gradients=True,
#         verbose=False,
#     )
    
#     # Create small test image (will be resized for model)
#     print("\nCreating 256x256 test image...")
#     test_image = Image.new("RGB", (256, 256), color=(128, 100, 150))
    
#     # Optimize at original 256x256 resolution
#     result = optimize_with_original_resolution(
#         pipeline,
#         test_image,
#         target_text="hello",
#         num_steps=5,
#         lr=0.01,
#     )
    
#     result.save("original_res_optimized.png")
#     print(f"\n✓ Saved 256x256 result to original_res_optimized.png")

class DummyRunner:
    def run(self, model_path: str, image_path: str, output_path: str):
        from deepseek_ocr_pipeline import DeepSeekOCRPipeline

        print("Loading model...")
        pipeline = DeepSeekOCRPipeline.from_pretrained(
            model_path,
            enable_image_gradients=True,
            use_unsloth=True,
            verbose=False,
        )

        # Create image.
        print("\nLoading test image...")
        test_image = Image.open(image_path).convert("RGB")

        # Run pixel optimization
        result = optimize_with_original_resolution(
            pipeline,
            test_image,
            target_text="cat",
            num_steps=10,
            lr=0.01,
        )

        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.save(output_path)
        print(f"\n✓ Saved to {output_path}")