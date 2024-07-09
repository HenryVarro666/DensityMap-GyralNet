'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-07-08 11:36:01
FilePath: /DensityMap+GNN/gradient_generate.py
'''
import os
import pyvista
import nibabel
import hcp_utils
import numpy as np
from tqdm import tqdm
import nibabel.freesurfer.io as fio
import argparse

def create_morph_data(data_path):
    """
    Creates morphological data for each subject in the given data directory.

    Returns:
        None
    """
    file_list = os.listdir(data_path)

    for file in file_list:
        # if '.withGrad.32k_fs_LR.Inner.vtk' not in file:
        if '.SpheSurf.RegByFS.Resp163842.vtk' not in file:
            continue
        if 'lh' in file:
            hemi = 'lh'
        elif 'rh' in file:
            hemi = 'rh'
        file_path = os.path.join(data_path, file)
        surf = pyvista.read(file_path)

        Gradient_Density = surf['gradient_density']

        file_path = os.path.join(data_path, "%s.grad"%(hemi))

        fio.write_morph_data(file_path, Gradient_Density, fnum=327680)
        print("Gradient Density has been written to ", file_path)

    return

def rescale_feature(data_path):
    """
    Rescales the 'sulc' and 'curv' features of each file in the specified data directory.

    Returns:
        None
    """
    # data_dir = "/home/lab/Documents/DensityMap-GyralNet/32k_3subjects"
    # file_list = os.listdir(data_dir)
    # pbar = tqdm(file_list)
    # for file in pbar:
    #     if '.withGrad.32k_fsaverage.flip.Sphere.vtk' not in file:

    file_list = os.listdir(data_path)

    for file in file_list:
        if '.SpheSurf.RegByFS.Resp163842.vtk' not in file:
            continue
        # subject_id = file.split('.')[0]
        hemi = None 
        if 'lh' in file:
            hemi = 'lh'
        elif 'rh' in file:
            hemi = 'rh'

        file_path = os.path.join(data_path, file)

        sphere = pyvista.read(file_path)
        grad = sphere['gradient_density']
        
        # # rescale sulc
        lower_bound = np.percentile(grad, 5)
        upper_bound = np.percentile(grad, 95)
        grad = np.clip(grad, lower_bound, upper_bound)


        # max-min normalization
        grad = (grad - np.min(grad)) / (np.max(grad) - np.min(grad))

        sphere['gradient_density'] = grad

        output_path = os.path.join(data_path, f"{hemi}.rescale.grad")
        fio.write_morph_data(output_path, grad, fnum=327680)    
        print("Rescaled Gradient Density has been written to ", output_path)
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gradient for subject")
    parser.add_argument("--subject_recon_dir", type=str, required=True, help="Subject ID")
    args = parser.parse_args()

    subject_recon_dir = args.subject_recon_dir
    data_path = subject_recon_dir
    
    create_morph_data(data_path)
    rescale_feature(data_path)