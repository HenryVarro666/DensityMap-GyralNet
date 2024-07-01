# # Part1
# import os

# def rename_files_in_recon_folders(parent_folder):
#     # 遍历指定文件夹中的所有子文件夹
#     for subdir in os.listdir(parent_folder):
#         subdir_path = os.path.join(parent_folder, subdir)
        
#         # 检查是否是文件夹
#         if os.path.isdir(subdir_path):
#             # 查找包含 '_recon' 的子文件夹
#             recon_folder = os.path.join(subdir_path, subdir + '_recon')
#             if os.path.exists(recon_folder):
#                 print(f"Processing folder: {recon_folder}")
                
#                 # 遍历 '_recon' 文件夹中的所有文件
#                 for filename in os.listdir(recon_folder):
#                     file_path = os.path.join(recon_folder, filename)
                    
#                     # 检查是否是文件
#                     if os.path.isfile(file_path):
#                         # 去除文件名第一个 '.' 及其之前的名字
#                         new_name = filename.split('.', 1)[-1]
                        
#                         # 如果去除后文件名以 'L' 开头则替换为 'lh'
#                         if new_name.startswith('L'):
#                             new_name = 'lh' + new_name[1:]
#                         # 如果去除后文件名以 'R' 开头则替换为 'rh'
#                         elif new_name.startswith('R'):
#                             new_name = 'rh' + new_name[1:]
                        
#                         new_file_path = os.path.join(recon_folder, new_name)
                        
#                         # 重命名文件
#                         os.rename(file_path, new_file_path)
#                         print(f"Renamed {filename} to {new_name}")

# # 使用示例
# parent_folder = './'
# rename_files_in_recon_folders(parent_folder)

####################################################################################################

# # Part2

# import os
# import shutil

# def move_files_to_surf_folder(parent_folder):
#     # 遍历指定文件夹中的所有子文件夹
#     for subdir in os.listdir(parent_folder):
#         subdir_path = os.path.join(parent_folder, subdir)
        
#         # 检查是否是文件夹
#         if os.path.isdir(subdir_path):
#             # 查找包含 '_recon' 的子文件夹
#             recon_folder = os.path.join(subdir_path, subdir + '_recon')
#             if os.path.exists(recon_folder):
#                 print(f"Processing folder: {recon_folder}")
                
#                 # 创建 'surf' 子文件夹
#                 surf_folder = os.path.join(recon_folder, 'surf')
#                 if not os.path.exists(surf_folder):
#                     os.makedirs(surf_folder)
#                     print(f"Created folder: {surf_folder}")
                
#                 # 遍历 '_recon' 文件夹中的所有文件
#                 for filename in os.listdir(recon_folder):
#                     file_path = os.path.join(recon_folder, filename)
                    
#                     # 检查是否是文件，并且不是 'surf' 文件夹本身
#                     if os.path.isfile(file_path):
#                         # 移动文件到 'surf' 文件夹
#                         shutil.move(file_path, os.path.join(surf_folder, filename))
#                         print(f"Moved {filename} to {surf_folder}")

# # 使用示例
# parent_folder = './'
# move_files_to_surf_folder(parent_folder)


####################################################################################################

# Part3

import os
import shutil

def delete_existing_files(parent_folder):
    # 遍历指定文件夹中的所有子文件夹
    for subdir in os.listdir(parent_folder):
        subdir_path = os.path.join(parent_folder, subdir)
        
        # 检查是否是文件夹
        if os.path.isdir(subdir_path):
            # 查找包含 '_recon' 的子文件夹
            recon_folder = os.path.join(subdir_path, subdir + '_recon')
            if os.path.exists(recon_folder):
                print(f"Processing folder for deletion: {recon_folder}")
                
                # 查找并删除文件
                for filename in os.listdir(recon_folder):
                    if filename.endswith('withGrad.164k_fsaverage.flip.rescale.Inner.vtk'):
                        file_path = os.path.join(recon_folder, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"Deleted {file_path}")

def copy_files_from_surf_to_upper_level(parent_folder):
    # 遍历指定文件夹中的所有子文件夹
    for subdir in os.listdir(parent_folder):
        subdir_path = os.path.join(parent_folder, subdir)
        
        # 检查是否是文件夹
        if os.path.isdir(subdir_path):
            # 查找包含 '_recon' 的子文件夹
            recon_folder = os.path.join(subdir_path, subdir + '_recon')
            if os.path.exists(recon_folder):
                print(f"Processing folder for copying: {recon_folder}")
                
                # 查找 'surf' 子文件夹
                surf_folder = os.path.join(recon_folder, 'surf')
                if os.path.exists(surf_folder):
                    # 遍历 'surf' 文件夹中的所有文件
                    for filename in os.listdir(surf_folder):
                        file_path = os.path.join(surf_folder, filename)
                        
                        # 检查文件名是否以指定字符串结尾
                        if filename.endswith('withGrad.164k_fsaverage.flip.Inner.vtk'):
                            # 复制文件到上上级目录
                            dest_path = os.path.join(subdir_path, filename)
                            shutil.copy(file_path, dest_path)
                            print(f"Copied {filename} to {subdir_path}")

# 使用示例
parent_folder = './'
delete_existing_files(parent_folder)
copy_files_from_surf_to_upper_level(parent_folder)

