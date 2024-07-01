'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-07-01 07:50:44
FilePath: /DensityMap+GNN/gradient_generate.py
'''
import os
import pyvista
import nibabel
import hcp_utils
import numpy as np
from tqdm import tqdm
from nibabel.freesurfer import io as fio

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

def create_morph_data():
    """
    Creates morphological data for each subject in the given data directory.

    Returns:
        None
    """
    data_dir = "./100206/100206_recon/surf"
    file_list = os.listdir(data_dir)

    for file in file_list:
        # if '.withGrad.32k_fs_LR.Inner.vtk' not in file:
        if '.withGrad.32k_fs_LR.Inner.vtk' not in file:
            continue
        if 'lh' in file:
            hemi = 'lh'
        elif 'rh' in file:
            hemi = 'rh'
        file_path = os.path.join(data_dir, file)
        surf = pyvista.read(file_path)
        Gradient_Density = surf['gradient_density']
        fio.write_morph_data(os.path.join(data_dir, "%s.grad.sulc"%(hemi)), Gradient_Density, fnum=327680)

    return


def create_morph_data_zero():
    """
    Creates morphological data for each subject in the given data directory.

    Returns:
        None
    """
    data_dir = "./100206/100206_recon/surf"
    file_list = os.listdir(data_dir)
    threshold = 0.7

    for file in file_list:
        # if '.withGrad.32k_fs_LR.Inner.vtk' not in file:
        if '.withGrad.32k_fs_LR.Inner.vtk' not in file:
            continue
        if 'lh' in file:
            hemi = 'lh'
        elif 'rh' in file:
            hemi = 'rh'
        file_path = os.path.join(data_dir, file)
        surf = pyvista.read(file_path)
        Gradient_Density = surf['gradient_density']
        Gradient_Density = Gradient_Density - threshold
        fio.write_morph_data(os.path.join(data_dir, "%s.%s.grad.sulc"%(hemi, threshold)), Gradient_Density, fnum=327680)

    return

def rescale_feature():
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

    data_dir = "./100206/100206_recon/surf"
    file_list = os.listdir(data_dir)

    for file in file_list:
        if '.withGrad.32k_fs_LR.Inner.vtk' not in file:
            continue
        subject_id = file.split('.')[0]
        if 'lh' in file:
            hemi = 'lh'
        elif 'lh' in file:
            hemi = 'rh'

        file_path = os.path.join(data_dir, file)

        sphere = pyvista.read(file_path)
        sulc = sphere['gradient_density']
        
        sulc = (sulc - np.min(sulc)) / (np.max(sulc) - np.min(sulc)) * (0.989 - (-0.4)) + (-0.4)

        sphere['gradient_density'] = sulc

        fio.write_morph_data(os.path.join(data_dir, "%s.grad.rescale.sulc"%(hemi)), sulc, fnum=327680)
        # sphere.save(file_path.replace('.withGrad.164k_fsaverage.flip.Sphere.vtk', '.withGrad.164k_fsaverage.flip.rescale.Sphere.vtk'), binary=False)
    return

if __name__ == "__main__":
    rescale_feature()
    # print("True")