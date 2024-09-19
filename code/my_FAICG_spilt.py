import os
import cv2
import numpy as np
import glob

path_FA = './All/FA/train/img'
# path_ICG = './All/ICG/train/img'
FA = '/train/img'
ICG = '/train/img'
# path = r'./ICG/val/img/*.png'
# path = r'./ICG/test/img/*.png'

# path = r'./unlabel_FA/*.jpg'

save_path = './DataLoader/data_splits/train.txt'

dirs_FA = os.listdir(path_FA)
# dirs_ICG = os.listdir(path_ICG)

for imgs in dirs_FA:
    FA_img = FA + '/' + str(imgs)
    FA_prod = FA_img.replace('img','mask')
    # ICG_img = ICG + '/' + str(imgs)
    # ICG_prod = ICG_img.replace('img','mask')
    # print(FA_img)
    # print(FA_prod)
    # print(ICG_img)
    # print(ICG_prod)
    f = open(save_path, 'a')
    # f.write(FA_img + ' ' + FA_prod + ' ' + ICG_img + ' ' + ICG_prod)
    f.write(FA_img + ' ' + FA_prod)
    f.write('\n')
    f.close()