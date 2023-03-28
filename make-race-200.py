import os
import shutil
import random

def copy_files(src_dir, dst_dir, num_files):
    # 获取源文件夹中的所有文件
    all_files = os.listdir(src_dir)

    # 从所有文件中随机选择 num_files 个文件
    selected_files = random.sample(all_files, num_files)

    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 复制选定的文件到目标文件夹
    for file in selected_files:
        src_file_path = os.path.join(src_dir, file)
        dst_file_path = os.path.join(dst_dir, file)
        shutil.copy(src_file_path, dst_file_path)

# 示例
source_dir = 'RACE/test'
destination_dir = 'RACE200'
number_of_files = 100
copy_files(os.path.join(source_dir, 'middle'), destination_dir, number_of_files)
copy_files(os.path.join(source_dir, 'high'), destination_dir, number_of_files)
