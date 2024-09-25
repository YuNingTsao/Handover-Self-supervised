import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from Utils.self_loss import *
from Utils.self_mask import *
from Utils.losses import *

class MaskedAutoencoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=128):
        super(MaskedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4,stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 28 * 28),  
            nn.ReLU(),
            nn.Unflatten(1, (256, 28, 28)),  
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
        )
        
    
    def forward(self, x):
        x , mask= apply_mask(x) # When to apply the mask
        # Encode the visible (unmasked) patches
        encoded = self.encoder(x)
        # Decode to reconstruct the full image (including the masked parts)
        decoded = self.decoder(encoded)
        # Only compute the loss on the masked regions
        mask_loss = masked_loss(decoded, x, mask)
#       outputs = {"self_pred": decoded}
        
        return mask_loss, decoded
    
class MaskedAutoencoderForSegmentation(nn.Module):
    def __init__(self, mask_ratio=0.75, pretrained_encoder=None, latent_dim=128, num_classes=2):
        super(MaskedAutoencoderForSegmentation, self).__init__()
        
        # Encoder (Compress image to latent space)
        self.encoder = pretrained_encoder.encoder
        
        # Decoder for Segmentation (Predict mask instead of reconstructing image)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 28 * 28),
            nn.ReLU(),
            nn.Unflatten(1, (256, 28, 28)),  # [B, 256, 28, 28]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, H/2, W/2]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, H/2, W/2]
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1),  # [B, 1, H, W] -> Segmentation Map
        )
        
        self.mask_ratio = mask_ratio
        self.diceloss = DiceLoss(num_classes)
    
    def forward(self, x, target):
        # Apply masking
        x_masked, mask = apply_mask(x)
        
        # Encode masked image to latent space
        encoded = self.encoder(x_masked)
        
        # Decode to get segmentation map
        segmentation_output = self.decoder(encoded)
#       loss = dice_loss(segmentation_output, target)
        loss = self.diceloss(torch.sigmoid(segmentation_output), torch.sigmoid(target[:].long()))

        return loss, segmentation_output
#       return None, segmentation_output