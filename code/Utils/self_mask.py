import torch
import numpy as np

def mask_image(image, mask_ratio=0.75):
    """
    Masks a given percentage of the image by setting certain patches to 0.
    Args:
        image: Input image tensor of shape (B, C, H, W)
        mask_ratio: Percentage of the image to mask (default: 75%)
    Returns:
        masked_image: Image with patches masked out
        mask: Boolean mask indicating which patches are masked
    """ 
    B, C, H, W = image.shape
    num_patches = H * W
    num_masked = int(mask_ratio * num_patches)
        
    # Flatten the image into patches
    flat_image = image.view(B, C, H * W)
        
    # Randomly choose which patches to mask
    mask = torch.zeros(B, H * W, dtype=torch.bool)
    for i in range(B):
        mask_indices = np.random.choice(H * W, num_masked, replace=False)
        mask[i, mask_indices] = True
        
    # Apply the mask (set masked patches to 0)
    masked_image = image.clone()
    masked_image.view(B, C, H * W)[mask] = 0
        
    return masked_image, mask

def apply_mask(x, mask_ratio=0.75):
    """
    Applies random masking to the input image.
    """
    B, C, H, W = x.shape
    num_patches = H * W
    num_masked = int(mask_ratio * num_patches)
        
    mask = torch.ones(B, C, H, W, device=x.device)
    for i in range(B):
        # Randomly select patches to mask
        mask_indices = np.random.choice(H * W, num_masked, replace=False)
        mask.view(B, C, H * W)[i, :, mask_indices] = 0
        mask = mask.bool()
        
    # Apply mask by setting masked regions to zero
    x_masked = x.clone()
    x_masked *= mask
        
    return x_masked, mask
