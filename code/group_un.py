import os
import shutil

source_directory = './All/unFA'  
target_directory = './All'

files = os.listdir(source_directory)

files.sort()
unfile_name = []

for file in files:
    unfile_name.append(file)
    # print(file)

print(len(unfile_name))

folders = {}
for name in unfile_name:
    folder_name = name.split("/")[-1].split(".")[0]
    last_underscore_index = folder_name.rfind('_')
    # 然後找到倒數第二個下劃線的位置
    second_last_underscore_index = folder_name.rfind('_', 0, last_underscore_index)
    # 截取到倒數第二個下劃線之前的部分
    result_string = folder_name[:second_last_underscore_index]
    if result_string not in folders:
        folders[result_string] = []
    folders[result_string].append(name)

# print(folders)
print(len(folders))
for folder, files in folders.items():
    # 確保每個目標資料夾都存在
    folder_path = os.path.join(target_directory, folder)
    os.makedirs(folder_path, exist_ok=True)
    
    for file_name in files:
        source_path = os.path.join(source_directory, file_name)  # 假設文件是 JPG 格式
        target_path = os.path.join(folder_path, file_name)
        
        # 移動文件
        shutil.copy(source_path, target_path)
        print(f"Copy {file_name} to {folder_path}")