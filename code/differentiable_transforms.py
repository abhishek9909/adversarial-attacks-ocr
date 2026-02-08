"""
Differentiable Image Transforms

Pure PyTorch implementations of image preprocessing operations that maintain
gradient flow. These replace PIL-based transforms for end-to-end differentiability.

All functions:
- Accept torch.Tensor inputs
- Return torch.Tensor outputs  
- Preserve requires_grad
- Use only differentiable operations
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union
import math


class DifferentiableNormalize(torch.nn.Module):
    """
    Differentiable image normalization.
    
    Normalizes tensor values from [0, 1] to normalized range using:
        output = (input - mean) / std
    """
    
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        super().__init__()
        # Register as buffers so they move with the module
        self.register_buffer('mean', torch.tensor(mean).view(3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(3, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [C, H, W] or [B, C, H, W] with values in [0, 1]
            
        Returns:
            Normalized tensor
        """
        # Handle both batched and unbatched input
        if x.dim() == 3:
            mean = self.mean.to(x.device, x.dtype)
            std = self.std.to(x.device, x.dtype)
        else:  # x.dim() == 4
            mean = self.mean.to(x.device, x.dtype).unsqueeze(0)
            std = self.std.to(x.device, x.dtype).unsqueeze(0)
        
        return (x - mean) / std
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse normalization (for visualization)."""
        if x.dim() == 3:
            mean = self.mean.to(x.device, x.dtype)
            std = self.std.to(x.device, x.dtype)
        else:
            mean = self.mean.to(x.device, x.dtype).unsqueeze(0)
            std = self.std.to(x.device, x.dtype).unsqueeze(0)
        
        return x * std + mean


def differentiable_resize(
    image: torch.Tensor,
    size: Tuple[int, int],
    mode: str = 'bilinear',
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Differentiable image resizing.
    
    Args:
        image: Tensor of shape [C, H, W] or [B, C, H, W]
        size: Target (height, width)
        mode: Interpolation mode ('bilinear', 'bicubic', 'nearest', 'area')
        align_corners: Whether to align corners (only for bilinear/bicubic)
        
    Returns:
        Resized tensor
    """
    needs_squeeze = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        needs_squeeze = True
    
    if mode in ('bilinear', 'bicubic'):
        resized = F.interpolate(
            image,
            size=size,
            mode=mode,
            align_corners=align_corners,
        )
    else:
        resized = F.interpolate(
            image,
            size=size,
            mode=mode,
        )
    
    if needs_squeeze:
        resized = resized.squeeze(0)
    
    return resized


def differentiable_center_pad(
    image: torch.Tensor,
    target_size: Tuple[int, int],
    fill_value: Union[float, Tuple[float, float, float]] = 0.5,
) -> torch.Tensor:
    """
    Pad image to target size, centering the original image.
    
    Args:
        image: Tensor of shape [C, H, W]
        target_size: Target (height, width)
        fill_value: Value(s) for padding (scalar or per-channel tuple)
        
    Returns:
        Padded tensor of shape [C, target_h, target_w]
    """
    C, H, W = image.shape
    target_h, target_w = target_size
    
    # Calculate padding
    pad_h = max(0, target_h - H)
    pad_w = max(0, target_w - W)
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Create output tensor filled with fill_value
    if isinstance(fill_value, (int, float)):
        output = torch.full(
            (C, target_h, target_w),
            fill_value,
            dtype=image.dtype,
            device=image.device,
        )
    else:
        # Per-channel fill
        fill_tensor = torch.tensor(fill_value, dtype=image.dtype, device=image.device)
        output = fill_tensor.view(C, 1, 1).expand(C, target_h, target_w).clone()
    
    # Place original image in center (this preserves gradients)
    # Handle case where image is larger than target
    src_top = max(0, -pad_top)
    src_left = max(0, -pad_left)
    src_bottom = H - max(0, H - (target_h - pad_top))
    src_right = W - max(0, W - (target_w - pad_left))
    
    dst_top = max(0, pad_top)
    dst_left = max(0, pad_left)
    dst_bottom = dst_top + (src_bottom - src_top)
    dst_right = dst_left + (src_right - src_left)
    
    output[:, dst_top:dst_bottom, dst_left:dst_right] = image[:, src_top:src_bottom, src_left:src_right]
    
    return output


def differentiable_crop(
    image: torch.Tensor,
    box: Tuple[int, int, int, int],
) -> torch.Tensor:
    """
    Crop image to specified box.
    
    Args:
        image: Tensor of shape [C, H, W]
        box: (left, top, right, bottom) - like PIL crop
        
    Returns:
        Cropped tensor
    """
    left, top, right, bottom = box
    return image[:, top:bottom, left:right].contiguous()


def differentiable_resize_and_pad(
    image: torch.Tensor,
    target_size: Tuple[int, int],
    fill_value: Union[float, Tuple[float, float, float]] = 0.5,
    mode: str = 'bilinear',
) -> torch.Tensor:
    """
    Resize image maintaining aspect ratio, then pad to target size.
    
    Args:
        image: Tensor of shape [C, H, W]
        target_size: Target (height, width)
        fill_value: Value(s) for padding
        mode: Interpolation mode
        
    Returns:
        Resized and padded tensor
    """
    C, H, W = image.shape
    target_h, target_w = target_size
    
    # Calculate scale to fit within target while maintaining aspect ratio
    scale = min(target_h / H, target_w / W)
    
    new_h = int(H * scale)
    new_w = int(W * scale)
    
    # Resize
    resized = differentiable_resize(image, (new_h, new_w), mode=mode)
    
    # Pad to target size
    padded = differentiable_center_pad(resized, target_size, fill_value)
    
    return padded


def calculate_dynamic_crop_grid(
    height: int,
    width: int,
    min_patches: int = 2,
    max_patches: int = 9,
    patch_size: int = 640,
) -> Tuple[int, int]:
    """
    Calculate optimal grid for cropping image into patches.
    
    Args:
        height: Image height
        width: Image width
        min_patches: Minimum number of patches
        max_patches: Maximum number of patches
        patch_size: Size of each patch
        
    Returns:
        (width_crops, height_crops) - number of crops in each dimension
    """
    aspect_ratio = width / height
    
    # Generate valid grid configurations
    target_ratios = set()
    for n in range(min_patches, max_patches + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if min_patches <= i * j <= max_patches:
                    target_ratios.add((i, j))
    
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    # Find best matching aspect ratio
    best_ratio = (1, 1)
    best_diff = float('inf')
    area = width * height
    
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_aspect)
        
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
        elif diff == best_diff:
            # Prefer configurations that use more of the image
            if area > 0.5 * patch_size * patch_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    
    return best_ratio


def dynamic_preprocess(
    image: torch.Tensor,
    min_patches: int = 2,
    max_patches: int = 9,
    patch_size: int = 640,
    include_thumbnail: bool = False,
) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
    """
    Dynamically preprocess image into patches based on aspect ratio.
    
    Args:
        image: Tensor of shape [C, H, W]
        min_patches: Minimum number of patches
        max_patches: Maximum number of patches
        patch_size: Size of each patch
        include_thumbnail: Whether to include a thumbnail of the full image
        
    Returns:
        (list of patch tensors, (width_crops, height_crops))
    """
    C, H, W = image.shape
    
    # Calculate optimal grid
    width_crops, height_crops = calculate_dynamic_crop_grid(
        H, W, min_patches, max_patches, patch_size
    )
    
    total_patches = width_crops * height_crops
    
    # Target dimensions for the grid
    target_w = patch_size * width_crops
    target_h = patch_size * height_crops
    
    # Resize image to fit grid
    resized = differentiable_resize(image, (target_h, target_w), mode='bilinear')
    
    # Extract patches
    patches = []
    for row in range(height_crops):
        for col in range(width_crops):
            top = row * patch_size
            left = col * patch_size
            bottom = top + patch_size
            right = left + patch_size
            
            patch = differentiable_crop(resized, (left, top, right, bottom))
            patches.append(patch)
    
    assert len(patches) == total_patches, f"Expected {total_patches} patches, got {len(patches)}"
    
    # Optionally add thumbnail
    if include_thumbnail and total_patches > 1:
        thumbnail = differentiable_resize(image, (patch_size, patch_size), mode='bilinear')
        patches.append(thumbnail)
    
    return patches, (width_crops, height_crops)


def pil_to_tensor(
    pil_image,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = True,
) -> torch.Tensor:
    """
    Convert PIL Image to differentiable PyTorch tensor.
    
    Args:
        pil_image: PIL Image (should be RGB mode)
        device: Target device
        dtype: Target dtype
        requires_grad: Whether to enable gradient computation
        
    Returns:
        Tensor of shape [C, H, W] with values in [0, 1]
    """
    import numpy as np
    
    # Ensure RGB
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy, then tensor
    np_array = np.array(pil_image, dtype=np.float32) / 255.0
    
    # [H, W, C] -> [C, H, W]
    tensor = torch.from_numpy(np_array).permute(2, 0, 1)
    tensor = tensor.to(device=device, dtype=dtype)
    
    if requires_grad:
        tensor = tensor.requires_grad_(True)
    
    return tensor


def tensor_to_pil(tensor: torch.Tensor):
    """
    Convert tensor back to PIL Image (for visualization).
    
    Args:
        tensor: Tensor of shape [C, H, W] with values in [0, 1]
        
    Returns:
        PIL Image
    """
    from PIL import Image
    import numpy as np
    
    # Detach and move to CPU
    tensor = tensor.detach().cpu()
    
    # Clamp to valid range
    tensor = tensor.clamp(0, 1)
    
    # [C, H, W] -> [H, W, C]
    np_array = tensor.permute(1, 2, 0).numpy()
    np_array = (np_array * 255).astype(np.uint8)
    
    return Image.fromarray(np_array, mode='RGB')


class DifferentiableImageProcessor:
    """
    Complete differentiable image processor for VLMs.
    
    Replaces PIL-based preprocessing with fully differentiable operations.
    """
    
    def __init__(
        self,
        patch_size: int = 640,
        base_size: int = 1024,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        use_dynamic_crops: bool = True,
        min_crops: int = 2,
        max_crops: int = 9,
    ):
        self.patch_size = patch_size
        self.base_size = base_size
        self.mean = mean
        self.std = std
        self.use_dynamic_crops = use_dynamic_crops
        self.min_crops = min_crops
        self.max_crops = max_crops
        
        self.normalize = DifferentiableNormalize(mean, std)
    
    def __call__(
        self,
        image: torch.Tensor,
        return_info: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, dict]]:
        """
        Process image into format expected by VLM.
        
        Args:
            image: Tensor of shape [C, H, W] with values in [0, 1]
            return_info: Whether to return additional info dict
            
        Returns:
            (local_patches, global_view) or (local_patches, global_view, info)
            - local_patches: Tensor of shape [N, C, patch_size, patch_size]
            - global_view: Tensor of shape [1, C, base_size, base_size]
        """
        C, H, W = image.shape
        device = image.device
        dtype = image.dtype
        
        info = {
            "original_size": (H, W),
            "crop_grid": (1, 1),
            "num_patches": 1,
        }
        
        # Process global view (always needed)
        global_view = differentiable_resize_and_pad(
            image,
            (self.base_size, self.base_size),
            fill_value=self.mean,
        )
        global_view = self.normalize(global_view)
        global_view = global_view.unsqueeze(0)  # Add batch dim
        
        # Process local patches if using dynamic crops
        if self.use_dynamic_crops and (W > self.patch_size or H > self.patch_size):
            patches, (w_crops, h_crops) = dynamic_preprocess(
                image,
                min_patches=self.min_crops,
                max_patches=self.max_crops,
                patch_size=self.patch_size,
                include_thumbnail=False,
            )
            
            # Normalize patches
            patches = [self.normalize(p) for p in patches]
            local_patches = torch.stack(patches, dim=0)
            
            info["crop_grid"] = (w_crops, h_crops)
            info["num_patches"] = len(patches)
        else:
            # No cropping needed - create dummy tensor
            local_patches = torch.zeros(
                (1, C, self.patch_size, self.patch_size),
                dtype=dtype,
                device=device,
            )
            info["crop_grid"] = (1, 1)
            info["num_patches"] = 0
        
        if return_info:
            return local_patches, global_view, info
        return local_patches, global_view


# Verification utilities
def verify_gradient_flow(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Verify that a tensor can receive gradients.
    
    Args:
        tensor: Tensor to check
        name: Name for logging
        
    Returns:
        True if gradients can flow
    """
    if not tensor.requires_grad:
        print(f"✗ {name}: requires_grad=False")
        return False
    
    if tensor.grad_fn is None and not tensor.is_leaf:
        print(f"✗ {name}: No grad_fn and not a leaf tensor")
        return False
    
    print(f"✓ {name}: Gradients enabled (leaf={tensor.is_leaf}, grad_fn={tensor.grad_fn})")
    return True


def test_gradient_flow():
    """Test that all transforms preserve gradient flow."""
    print("Testing gradient flow through transforms...")
    
    # Create test image with gradients
    image = torch.randn(3, 256, 256, requires_grad=True)
    
    # Test each transform
    tests_passed = 0
    tests_total = 0
    
    # Test resize
    tests_total += 1
    resized = differentiable_resize(image, (128, 128))
    if resized.requires_grad:
        loss = resized.sum()
        loss.backward()
        if image.grad is not None:
            print("✓ differentiable_resize: Gradients flow correctly")
            tests_passed += 1
        else:
            print("✗ differentiable_resize: No gradient in input")
    else:
        print("✗ differentiable_resize: Output doesn't require grad")
    
    # Reset gradient
    image.grad = None
    
    # Test padding
    tests_total += 1
    padded = differentiable_center_pad(image, (300, 300))
    if padded.requires_grad:
        loss = padded.sum()
        loss.backward()
        if image.grad is not None:
            print("✓ differentiable_center_pad: Gradients flow correctly")
            tests_passed += 1
        else:
            print("✗ differentiable_center_pad: No gradient in input")
    else:
        print("✗ differentiable_center_pad: Output doesn't require grad")
    
    # Reset gradient
    image.grad = None
    
    # Test crop
    tests_total += 1
    cropped = differentiable_crop(image, (10, 10, 100, 100))
    if cropped.requires_grad:
        loss = cropped.sum()
        loss.backward()
        if image.grad is not None:
            print("✓ differentiable_crop: Gradients flow correctly")
            tests_passed += 1
        else:
            print("✗ differentiable_crop: No gradient in input")
    else:
        print("✗ differentiable_crop: Output doesn't require grad")
    
    # Reset and test normalization
    image = torch.randn(3, 256, 256, requires_grad=True)
    tests_total += 1
    normalizer = DifferentiableNormalize()
    normalized = normalizer(image)
    if normalized.requires_grad:
        loss = normalized.sum()
        loss.backward()
        if image.grad is not None:
            print("✓ DifferentiableNormalize: Gradients flow correctly")
            tests_passed += 1
        else:
            print("✗ DifferentiableNormalize: No gradient in input")
    else:
        print("✗ DifferentiableNormalize: Output doesn't require grad")
    
    # Test full processor
    image = torch.randn(3, 800, 600, requires_grad=True)
    tests_total += 1
    processor = DifferentiableImageProcessor()
    local_patches, global_view = processor(image)
    if global_view.requires_grad:
        loss = global_view.sum()
        loss.backward()
        if image.grad is not None:
            print("✓ DifferentiableImageProcessor: Gradients flow correctly")
            tests_passed += 1
        else:
            print("✗ DifferentiableImageProcessor: No gradient in input")
    else:
        print("✗ DifferentiableImageProcessor: Output doesn't require grad")
    
    print(f"\nResults: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


if __name__ == "__main__":
    test_gradient_flow()
