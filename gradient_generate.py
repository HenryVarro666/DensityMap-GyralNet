'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-07-07 12:16:00
FilePath: /DensityMap+GNN/gradient_generate.py
'''
import os
import pyvista
import nibabel
import hcp_utils
import numpy as np
from tqdm import tqdm
import nibabel.freesurfer.io as fio

def write_vtk():
    """
    This function reads geometry and morphological data from files and saves it in a VTK file format.
    """
    points, faces = fio.read_geometry('/media/jialec/DATA/HCP_download_for_weiyan/100206_3T_Structural_preproc_extended/100206/T1w/100206/surf/lh.sphere')
    sulc = fio.read_morph_data('/media/jialec/DATA/HCP_download_for_weiyan/100206_3T_Structural_preproc_extended/100206/T1w/100206/surf/lh.sulc')
    curv = fio.read_morph_data('/media/jialec/DATA/HCP_download_for_weiyan/100206_3T_Structural_preproc_extended/100206/T1w/100206/surf/lh.curv')
    thickness = fio.read_morph_data('/media/jialec/DATA/HCP_download_for_weiyan/100206_3T_Structural_preproc_extended/100206/T1w/100206/surf/lh.thickness')

    # points, faces = fio.read_geometry('/usr/local/freesurfer/7.2.0/subjects/fsaverage/surf/lh.sphere')
    sulc = fio.read_morph_data('/usr/local/freesurfer/7.2.0/subjects/fsaverage/surf/lh.sulc')
    curv = fio.read_morph_data('/usr/local/freesurfer/7.2.0/subjects/fsaverage/surf/lh.curv')
    # par_fs = fio.read_annot('/usr/local/freesurfer/7.2.0/subjects/fsaverage/label/lh.aparc.a2005s.annot')
    # par_fs_2009 = fio.read_annot('/usr/local/freesurfer/7.2.0/subjects/fsaverage/label/lh.aparc.a2009s.annot')

    faces = np.concatenate((np.ones((faces.shape[0], 1))*3, faces), axis=-1).astype(np.int32)
    surf = pyvista.PolyData(points, faces)
    surf['sulc'] = sulc
    surf['curv'] = curv
    surf['thickness'] = thickness
    # surf['par_fs'] = par_fs
    # surf['par_fs_2009'] = par_fs_2009
    # surf.save('./delete_this.vtk', binary=False)
    surf.save('./delete_this_hcp.fsaverage.vtk', binary=False)
    print("True")
    return

def create_morph_data(data_path):
    """
    Creates morphological data for each subject in the given data directory.

    Returns:
        None
    """
    file_list = os.listdir(data_path)

    for file in file_list:
        # if '.withGrad.32k_fs_LR.Inner.vtk' not in file:
        if '.withGrad.32k_fs_LR.Inner.vtk' not in file:
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

def create_morph_data_zero(data_path):
    """
    Creates morphological data for each subject in the given data directory.

    Returns:
        None
    """
    file_list = os.listdir(data_path)
    threshold = 0.3

    for file in file_list:
        # if '.withGrad.32k_fs_LR.Inner.vtk' not in file:
        if '.withGrad.32k_fs_LR.Inner.vtk' not in file:
            continue
        if 'lh' in file:
            hemi = 'lh'
        elif 'rh' in file:
            hemi = 'rh'
        file_path = os.path.join(data_path, file)
        surf = pyvista.read(file_path)
        Gradient_Density = surf['gradient_density']
        Gradient_Density = Gradient_Density - threshold
        print(data_path)
        io.write_morph_data(os.path.join(data_path, "%s.%s.grad.sulc"%(hemi, threshold)), Gradient_Density, fnum=327680)

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
        if '.withGrad.32k_fs_LR.Inner.vtk' not in file:
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
        # sphere.save(file_path.replace('.withGrad.164k_fsaverage.flip.Sphere.vtk', '.withGrad.164k_fsaverage.flip.rescale.Sphere.vtk'), binary=False)
        print("Rescaled Gradient Density has been written to ", output_path)
    
    return

def flip_feature():
    """
    Flips the 'sulc' and 'curv' features in VTK files and saves the flipped data.

    This function reads VTK files from a specified directory, flips the 'sulc' and 'curv' features,
    and saves the flipped data to new files. The function assumes that the input VTK files have a
    specific naming convention and directory structure.

    Returns:
        None
    """
    data_dir = "/work/users/j/i/jialec/For_Caochao/HCP_data"
    file_list = os.listdir(data_dir)
    pbar = tqdm(file_list)
    for file in pbar:
        if '.withGrad.164k_fsaverage.Sphere.vtk' not in file:
            continue
        subject_id = file.split('.')[0]
        if '.L.' in file:
            hemi = 'lh'
        elif '.R.' in file:
            hemi = 'rh'

        file_path = os.path.join(data_dir, file)

        sphere = pyvista.read(file_path)
        sulc = -sphere['sulc']
        curv = -sphere['curv']

        sphere['sulc'] = sulc
        sphere['curv'] = curv

        io.write_morph_data(os.path.join(data_dir, "%s.%s.flip.sulc"%(subject_id, hemi)), sulc, fnum=327680)
        io.write_morph_data(os.path.join(data_dir, "%s.%s.flip.curv"%(subject_id, hemi)), curv, fnum=327680)
        sphere.save(file_path.replace('.withGrad.164k_fsaverage.Sphere.vtk', '.withGrad.164k_fsaverage.flip.Sphere.vtk'), binary=False)
    return

if __name__ == "__main__":
    data_path = "/mnt/d/DensityMap-GyralNet/32k_3subjects/100206/100206_recon/surf"
    rescale_feature(data_path)
    # print("True")
    create_morph_data(data_path)