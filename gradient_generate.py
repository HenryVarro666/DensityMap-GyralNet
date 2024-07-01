'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-06-25 23:06:18
FilePath: /DensityMap+GNN/gradient_generate.py
'''
import os
import pyvista
import nibabel
import hcp_utils
import numpy as np
from tqdm import tqdm
from nibabel.freesurfer import io as fio

def delete_this():
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


if __name__ == "__main__":
    delete_this()

    # print("True")