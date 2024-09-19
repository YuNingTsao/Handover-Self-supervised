import torch.nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

import numpy as np
import os
import cv2

def near_mixup_data(inputs, alpha = 1, r = True):
    if alpha > 0:
        lamb = np.random.beta(alpha, alpha)
    else:
        lamb = 1
    file_imgs = len(inputs)
    mixed_inputs = []
    mixed_inputs.append(inputs[0])
    for n in range(file_imgs-1):    
        mixed = lamb * inputs[n+1] + (1 - lamb) * inputs[n]
        mixed_inputs.append(mixed)

    return mixed_inputs

def range_mixup_data(inputs, alpha = 1, r = True):
    if alpha > 0:
        lamb = np.random.beta(alpha, alpha)
    else:
        lamb = 1
    file_imgs = len(inputs)
    half_file_imgs = int(file_imgs/2)
    mixed_inputs = []
    for n in range(half_file_imgs):
        mixed_inputs.append(inputs[n])
    for n in range(file_imgs - half_file_imgs):    
        mixed = lamb * inputs[n+half_file_imgs] + (1 - lamb) * inputs[n]
        mixed_inputs.append(mixed)

    return mixed_inputs

def main():
    source_directory = './All/Group_unFA' 
    near_target_directory = './All/near_unFA'
    range_target_directory = './All/range_unFA'

    files = os.listdir(source_directory)
    files.sort()
    to_tensor = transforms.ToTensor()
    total = 0
    for file in files:
        file_path = os.path.join(source_directory, file)
        imgs = os.listdir(file_path)
        imgs.sort()
        images_list = []
        for img in imgs:
            image_path = os.path.join(file_path, img)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            torch_image = to_tensor(Image.fromarray(np.uint8(image)))
            # print(torch_image.shape)
            images_list.append(torch_image)
    #     print(f'{file} has {len(images_list)} images')
    #     total += len(images_list)
        # for i in range(len(images_list)):
        #     os.makedirs(target_directory, exist_ok=True)
        #     print(images_list[i].shape)
        #     unimage = transforms.ToPILImage()(images_list[i])
        #     unimage.save(os.path.join(target_directory, imgs[i]))
        near_mixed_up = near_mixup_data(images_list, alpha = 1)
        for i in range(len(near_mixed_up)):
            os.makedirs(near_target_directory, exist_ok=True)
            # print(mixed_up[i].shape)
            unimage = transforms.ToPILImage()(near_mixed_up[i])
            unimage.save(os.path.join(near_target_directory, imgs[i]))
        print(f'near mixup {file} has {len(near_mixed_up)} images')
        range_mixed_up = range_mixup_data(images_list, alpha = 1)
        for i in range(len(range_mixed_up)):
            os.makedirs(range_target_directory, exist_ok=True)
            unimage = transforms.ToPILImage()(range_mixed_up[i])
            unimage.save(os.path.join(range_target_directory, imgs[i]))
        print(f'range mixup {file} has {len(range_mixed_up)} images')
        # input()
    #     total += len(range_mixed_up)
    # print(total)

if __name__ == "__main__":
    main()