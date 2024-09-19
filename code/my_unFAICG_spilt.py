import os
import random

path_FA = './All/unFA'
path_ICG = './All/unICG'

save_path = './DataLoader/data_splits/All/unlabel.txt'

unFA_files = os.listdir(path_FA)
unICG_files = os.listdir(path_ICG)

file_mapping = {}

for file_name in unFA_files:
    file_name_part = file_name.split('_', 1)[1]
    # print(file_name_part)
    file_name_part = "ICG_" + file_name_part
    if file_name_part in unICG_files:
        file_mapping[file_name] = file_name_part

unmapped_FA_files = [file_name for file_name in unFA_files if file_name not in file_mapping.keys()]
unmapped_ICG_files = [file_name for file_name in unICG_files if file_name not in file_mapping.values()]

for file_name in unmapped_FA_files:
    file_name_part = file_name.split('_', 1)[1]
    file_name_part = "ICG_" + file_name_part
    if file_name_part not in unmapped_ICG_files:
        file_mapping[file_name] = random.choice(unmapped_ICG_files)

# print(file_mapping)
with open(save_path, 'a') as f:
    for file_name in file_mapping:
        f.write(f"{'/unFA/' + file_name} {'/unICG/' + file_mapping[file_name]}\n")