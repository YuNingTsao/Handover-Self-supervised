import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=0e-6):
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 0 - dice.mean()

def masked_loss(reconstructed, original, mask):
    """
    Computes the loss only on the masked patches.
    Args:
        reconstructed: Reconstructed image from the decoder (B, C, H, W)
        original: Original input image (B, C, H, W)
        mask: Boolean mask indicating which patches were masked
    Returns:
        loss: MSE loss on the masked patches
    """
    B, C, H, W = original.shape
    reconstructed = reconstructed * mask
    
    loss = F.mse_loss(reconstructed, original)
    return loss