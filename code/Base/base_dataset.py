import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
import os


class BaseDataSet(Dataset):
    def __init__(self, split, val=False):
        self.split = split
        self.val = val

        self.to_tensor = transforms.ToTensor()
        
    def __getitem__(self, index):
        if self.val:
            image, label, image_id = self._load_data(index)
            image = cv2.resize(image,(224,224), interpolation=cv2.INTER_NEAREST)
            image = self.to_tensor(Image.fromarray(np.uint8(image)))
            label_val = cv2.resize(label.astype(float),(224,224), interpolation=cv2.INTER_NEAREST )
            label_val = torch.from_numpy(np.array(label_val, dtype=np.int32)).long()

            return image, label_val, image_id
        elif self.split == "train_supervised":
            image_FA, label_FA, image_FAid, image_ICG, label_ICG, image_ICGid = self._load_data(index)
            
            assert image_FAid == image_ICGid

            img_FA = cv2.resize(image_FA,(224,224), interpolation=cv2.INTER_NEAREST)
            label_FA = cv2.resize(label_FA.astype(float),(224,224), interpolation=cv2.INTER_NEAREST )
            img_ICG = cv2.resize(image_ICG,(224,224), interpolation=cv2.INTER_NEAREST)
            label_ICG = cv2.resize(label_ICG.astype(float),(224,224), interpolation=cv2.INTER_NEAREST )
        
            image_FA = self.to_tensor(Image.fromarray(np.uint8(img_FA)))
            image_ICG = self.to_tensor(Image.fromarray(np.uint8(img_ICG)))

            label_FA = torch.from_numpy(np.array(label_FA, dtype=np.int32)).long()
            label_ICG = torch.from_numpy(np.array(label_ICG, dtype=np.int32)).long()

            return image_FA, image_ICG, label_ICG, image_ICGid
        elif self.split == "train_unsupervised":
            image, label, image_id = self._load_data(index)
            img = cv2.resize(image,(224,224), interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label.astype(float),(224,224), interpolation=cv2.INTER_NEAREST )

            image = self.to_tensor(Image.fromarray(np.uint8(img)))
            label = torch.from_numpy(np.array(label, dtype=np.int32)).long()

            return image, label, image_id
        else:
            return None
