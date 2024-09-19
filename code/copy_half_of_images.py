import os
import shutil
import random

def copy_images(source_folder, dest_folder, fraction):
    # 確保目標資料夾存在
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # 獲取所有圖片檔案
    images = [file for file in os.listdir(source_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_count = int(len(images) * fraction)
    
    # 隨機選擇指定比例的圖片
    selected_images = random.sample(images, selected_count)
    
    # 複製圖片到目標資料夾
    for image in selected_images:
        src_image_path = os.path.join(source_folder, image)
        dest_image_path = os.path.join(dest_folder, image)
        shutil.copy(src_image_path, dest_image_path)
        print(f"Copied {image} to {dest_folder}")

# 使用函式
source_folder = './All/range_unFA'  # 源資料夾路徑
copy_images(source_folder, './All/Half_range_unFA', 0.5)  # 將一半的圖片複製到B資料夾
copy_images(source_folder, './All/Quarter_range_unFA', 0.25) # 將四分之一的圖片複製到C資料夾
