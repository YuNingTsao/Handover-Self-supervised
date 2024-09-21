import torch
import torch.nn as nn
import torch.nn.functinal as F
import numpy as np
from functools import partial

    
class MaskedAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super(MaskedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid to bring values between 0 and 1
        )
        
    def mask_image(self, image, mask_ratio=0.75):
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
        loss = F.mse_loss(reconstructed[mask], original[mask])
        return loss
    
    def forward(self, x):
        x = mask_image(x) # When to apply the mask
        # Encode the visible (unmasked) patches
        encoded = self.encoder(x)
        # Decode to reconstruct the full image (including the masked parts)
        decoded = self.decoder(encoded)
        # Only compute the loss on the masked regions
        mask_loss = self.masked_loss(decoded, x, mask)
        outputs = {'self_pred': decoded}
        
        return mask_loss, decoded
    
class MaskedAutoencoderForSegmentation(nn.Module):
    def __init__(self, input_size=(256, 256), mask_ratio=0.75, pretrained_encoder=None):
        super(MaskedAutoencoderForSegmentation, self).__init__()
        
        # Encoder (Compress image to latent space)
        self.encoder = pretrained_encoder.encoder
        
        # Decoder for Segmentation (Predict mask instead of reconstructing image)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, H/4, W/4]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, H/2, W/2]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # [B, 1, H, W] -> Segmentation Map
            nn.Sigmoid()  # Sigmoid for binary segmentation, or softmax for multi-class
        )
        
        self.mask_ratio = mask_ratio
    
    
    def apply_mask(self, x):
        """
        Applies random masking to the input image.
        """
        B, C, H, W = x.shape
        num_patches = H * W
        num_masked = int(self.mask_ratio * num_patches)
        
        mask = torch.ones(B, H, W, device=x.device)
        for i in range(B):
            # Randomly select patches to mask
            mask_indices = np.random.choice(H * W, num_masked, replace=False)
            mask.view(B, H * W)[i, mask_indices] = 0
        
        # Apply mask by setting masked regions to zero
        x_masked = x.clone()
        x_masked *= mask.unsqueeze(1)  # Apply mask across the channel dimension
        
        return x_masked, mask
    
    def dice_loss(pred, target, smooth=1e-6):
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def forward(self, x, target):
        # Apply masking
        x_masked, mask = self.apply_mask(x)
        
        # Encode masked image to latent space
        encoded = self.encoder(x_masked)
        
        # Decode to get segmentation map
        segmentation_output = self.decoder(encoded)
        loss = self.dice_loss(segmentation_output, target)
        
        return loss, segmentation_output