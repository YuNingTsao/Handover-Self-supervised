from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2
import os
import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

class BasicDataset(Dataset):
    def __init__(self, num_classes = 2, data_dir=None, choose=None,split=None):
        super(BasicDataset, self).__init__()
        self.num_classes = num_classes
        self.root = choose + "/" + data_dir
        self.choose = choose
        self.split = split

        self.to_tensor = transforms.ToTensor()

        self.files = []
        self.ICGfiles = []
        self._set_files()

    def _set_files(self):
        self.root = os.path.join(self.root)
        prefix = "DataLoader/data_splits" + "/" + self.choose

        if self.split == "val":
            file_list = os.path.join(prefix, "FA_val.txt")
            file_list = [line.rstrip().split(' ') for line in open(file_list, "r")]
            self.files, self.labels = list(zip(*file_list))
        elif self.split == "test":
            file_list = os.path.join(prefix, "FA_test.txt")
            file_list = [line.rstrip().split(' ') for line in open(file_list, "r")]
            self.files, self.labels = list(zip(*file_list))
        elif self.split == "train_supervised":
            file_list = os.path.join(prefix, "FAICG_train.txt")
            file_list = [line.rstrip().split(' ') for line in open(file_list, "r")]
            self.files, self.labels, self.ICGfiles, self.ICGlabels = list(zip(*file_list))
        
        elif self.split == "train_unsupervised":
            file_list = os.path.join(prefix, "unlabel.txt")
            file_list = [(line.rstrip(), 'None') for line in open(file_list, "r")]
            self.files, self.labels = list(zip(*file_list))
        else:
            raise ValueError(f"Invalid split name {self.split}")

    def _load_data(self, index):
        threshold = 0
        if self.split == "train_supervised":
            image_FApath = os.path.join(self.root, self.files[index][1:])
            image_FA = np.asarray(Image.open(image_FApath), dtype=np.float32)
            # image_FA = image_FA / 255.0
            # image_FA = image_FA.astype(np.float32)
            image_FAid = self.files[index].split("/")[-1].split(".")[0]

            label_FApath = os.path.join(self.root, self.labels[index][1:])
            label_FA = np.asarray(Image.open(label_FApath).convert('L'), dtype=np.int32)
            # label_FA[label_FA > threshold] = 1
            # label_FA[label_FA <= threshold] = 0
            # label_FA = label_FA.astype(np.int32)
            
            image_ICGpath = os.path.join(self.root, self.ICGfiles[index][1:])
            image_ICG = np.asarray(Image.open(image_ICGpath), dtype=np.float32)
            # image_ICG = image_ICG / 255.0
            # image_ICG = image_ICG.astype(np.float32)
            image_ICGid = self.ICGfiles[index].split("/")[-1].split(".")[0]

            label_ICGpath = os.path.join(self.root, self.ICGlabels[index][1:])
            label_ICG = np.asarray(Image.open(label_ICGpath).convert('L'), dtype=np.int32)
            # label_ICG[label_ICG > threshold] = 1
            # label_ICG[label_ICG <= threshold] = 0
            # label_ICG = label_ICG.astype(np.int32)

            assert image_FAid == image_ICGid, 'FA {} & ICG {} names do not match'.format(image_FAid, image_ICGid)
            
            return image_FA, label_FA, image_FAid, image_ICG, label_ICG, image_ICGid
        elif self.split == "train_unsupervised":
            image_path = os.path.join(self.root, self.files[index][1:])
            image = np.asarray(Image.open(image_path), dtype=np.float32)
            image_id = self.files[index].split("/")[-1].split(".")[0]
        
            label = np.zeros(image.shape[:2])

            return image, label, image_id
        elif self.split == "val":
            image_path = os.path.join(self.root, self.files[index][1:])
            image = np.asarray(Image.open(image_path), dtype=np.float32)
            image_id = self.files[index].split("/")[-1].split(".")[0]
            label_path = os.path.join(self.root, self.labels[index][1:])
            label = np.asarray(Image.open(label_path).convert('L'), dtype=np.int32)

            return image, label, image_id
        elif self.split == "test":
            image_path = os.path.join(self.root, self.files[index][1:])
            image = np.asarray(Image.open(image_path), dtype=np.float32)
            image_id = self.files[index].split("/")[-1].split(".")[0]
            label_path = os.path.join(self.root, self.labels[index][1:])
            label = np.asarray(Image.open(label_path).convert('L'), dtype=np.int32)

            return image, label, image_id
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.split == "val":
            image, label, image_id = self._load_data(index)
            image = cv2.resize(image,(224,224), interpolation=cv2.INTER_NEAREST)
            image = self.to_tensor(Image.fromarray(np.uint8(image)))
            label_val = cv2.resize(label.astype(float),(224,224), interpolation=cv2.INTER_NEAREST )
            label_val = torch.from_numpy(np.array(label_val, dtype=np.int32)).long()

            return image, label_val, image_id
        elif self.split == "test":
            image, label, image_id = self._load_data(index)
            image = cv2.resize(image,(224,224), interpolation=cv2.INTER_NEAREST)
            image = self.to_tensor(Image.fromarray(np.uint8(image)))
            label_test = cv2.resize(label.astype(float),(224,224), interpolation=cv2.INTER_NEAREST )
            label_test = torch.from_numpy(np.array(label_test, dtype=np.int32)).long()

            return image, label_test, image_id
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