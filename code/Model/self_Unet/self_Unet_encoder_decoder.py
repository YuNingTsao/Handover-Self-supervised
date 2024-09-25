import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from Utils.self_loss import *
from Utils.self_mask import *
from Utils.losses import *
from Model.self_Unet.Unet_parts import *

class SequentialWithSave(nn.Sequential):
    def __init__(self, *args):
        super(SequentialWithSave, self).__init__(*args)
        self.outputs = []

    def forward(self, x):
        self.outputs = []  # Reset outputs
        for layer in self:
            x = layer(x)
            self.outputs.append(x)
        return x

class MaskedAutoencoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, bilinear=False):
        super(MaskedAutoencoder, self).__init__()
        self.sequential = SequentialWithSave()
        if bilinear:
            factor = 2
        else:
            factor = 1
            

        self.encoder = self.sequential(
            DoubleConv(in_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024 // factor)
       )
            
        self.up1 = Up(1024, 512 // factor, bilinear),
        self.up2 = Up(512, 256, bilinear),
        self.up3 = Up(256, 128, bilinear),
        self.up4 = Up(128, 64, bilinear),
        self.outc = OutConv(64, out_channels)
        
    
    def forward(self, x):
        x , mask= apply_mask(x) # When to apply the mask
        # Encode the visible (unmasked) patches
        encoded = self.encoder(x)
        # Decode to reconstruct the full image (including the masked parts)
        x5 = self.sequential.outputs.pop()
        x4 = self.sequential.outputs.pop()
        x3 = self.sequential.outputs.pop()
        x2 = self.sequential.outputs.pop()
        x1 = self.sequential.outputs.pop()
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        decoded = self.outc(x)
        # Only compute the loss on the masked regions
        mask_loss = masked_loss(decoded, x, mask)
#       outputs = {"self_pred": decoded}
        
        return mask_loss, decoded
    
class MaskedAutoencoderForSegmentation(nn.Module):
    def __init__(self, mask_ratio=0.75, pretrained_encoder=None, num_classes=2, bilinear=False):
        super(MaskedAutoencoderForSegmentation, self).__init__()
        self.sequential = SequentialWithSave()
        if bilinear:
            factor = 2
        else:
            factor = 1
            
        # Encoder (Compress image to latent space)
        self.encoder = pretrained_encoder.encoder
        
        # Decoder for Segmentation (Predict mask instead of reconstructing image)
        self.up1 = Up(1024, 512 // factor, bilinear),
        self.up2 = Up(512, 256, bilinear),
        self.up3 = Up(256, 128, bilinear),
        self.up4 = Up(128, 64, bilinear),
        self.outc = OutConv(64, num_classes)
       
        
        self.mask_ratio = mask_ratio
        self.diceloss = DiceLoss(num_classes)
    
    def forward(self, x, target):
        # Apply masking
        x_masked, mask = apply_mask(x)
        
        # Encode masked image to latent space
        encoded = self.encoder(x_masked)
        # Decode to get segmentation map
        x5 = self.sequential.outputs.pop()
        x4 = self.sequential.outputs.pop()
        x3 = self.sequential.outputs.pop()
        x2 = self.sequential.outputs.pop()
        x1 = self.sequential.outputs.pop()
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        segmentation_output = self.outc(x)
        
        #       loss = dice_loss(segmentation_output, target)
        loss = self.diceloss(torch.sigmoid(segmentation_output), torch.sigmoid(target[:].long()))

        return loss, segmentation_output
#       return None, segmentation_output