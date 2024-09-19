import os
import cv2
import numpy as np
import glob

# path = r'./ICG/train/img/*.png'
# path = r'./All/FA/val/img/*.png'
# path = r'./FA/test/img/*.png'

# path = r'./All/unlabel_FA/*.png'
# path = r'./All/unlabel_FA/*.jpg'

# path = r'./All/unFA/*.jpg'
# path = r'./All/near_unFA/*.jpg'
# path = r'./All/range_unFA/*.jpg'


# path = r'./All/Half_range_unFA/*.jpg'
path = r'./All/Quarter_range_unFA/*.jpg'

# path = r'./All/unlabel_ICG/*.jpg'

save_path = r'./DataLoader/data_splits/'


def showalldataname():
    for i,img_path in enumerate(glob.glob(path)):
        # print(img_path)
        
        img_filename = img_path[:-4]
        print(img_filename)
        # img_a,img_b,img_c,img_d,img_e, img_f=img_filename.split('/')
        img_a,img_b,img_c,img_d=img_filename.split('/')
        # print(img_f)
        # list_path = save_path + img_c + '_' + img_d +'.txt'
        # list_path = save_path + img_b + '.txt'
        # list_path = save_path + 'unlabel_FA.txt'
        
        # list_path = save_path + 'un.txt'
        # list_path = save_path + 'unnear.txt'
        # list_path = save_path + 'unrange.txt'
        # list_path = save_path + 'Half_unlabel.txt'
        list_path = save_path + 'Quarter_unlabel.txt'

        # print(list_path)
        
        # _img = '/'+img_c+'/'+img_d+'/'+img_e+'.png'
        _img = '/'+img_d+'.jpg'
        # _img = '/'+img_c+'.png'
        print(_img)

        # label_path = img_path.replace('img','mask')
        # # print(label_path)
        # label_filename = label_path[:-4]
        
        # label_a,label_b,label_c,label_d,label_e=label_filename.split('/')
  
        # _label = '/'+label_c+'/'+label_d+'/'+label_e+'.png'
        # # print(_label)
        
        f = open(list_path, 'a')
        # f.write(_img+' '+_label)
        # f.write(_img+' '+'None')
        f.write(_img)
        f.write('\n')
        f.close()

showalldataname()