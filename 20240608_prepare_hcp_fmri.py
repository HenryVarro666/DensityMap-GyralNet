#### This code is used for prepare the data for cao chao.
#### He need to use the well-aligned HCP structural and fmri gradient density data.
#### I suppose he need to use the data in the fsaverage space.
#### I have previously transformed the fmri gradient density data from fs_LR 32k space to fsaverage space.
#### Therefore, in this code, I have to create the fs_LR_164k raw structural data into vtk format and transform it into fsaverage space.

import os
import pyvista
import nibabel
import hcp_utils
import numpy as np
from tqdm import tqdm
from nibabel.freesurfer import io as fio

def delete_this():
    """
    This function reads geometry and morph data from files and saves it in a VTK file format.
    It performs the following steps:
    1. Reads points and faces from a geometry file.
    2. Reads sulc, curv, and thickness data from morph files.
    3. Creates a PolyData object using the points and faces.
    4. Assigns sulc, curv, and thickness data to the PolyData object.
    5. Saves the PolyData object in VTK file format.

    Returns:
    - None
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

def create_vtk():
    """
    Create VTK files for each subject in the given data directory.

    This function loads various files for each subject, processes the data, and saves the resulting VTK files.

    Returns:
        None
    """
    data_dir = "/media/jialec/My Book/DATA/For_Caochao/Raw_data/"
    subject_list = os.listdir(data_dir)
    pbar = tqdm(subject_list)
    for subject in pbar:
        if '.zip' in subject or '.md5' in subject:
            continue
        subject_dir = os.path.join(data_dir, subject)
        subject_id = subject.split('_')[0]

        file_list = ['{}.L.sphere.32k_fs_LR.surf.gii'.format(subject_id),
                     '{}.R.sphere.32k_fs_LR.surf.gii'.format(subject_id),
                     '{}.L.white_MSMAll.32k_fs_LR.surf.gii'.format(subject_id),
                     '{}.R.white_MSMAll.32k_fs_LR.surf.gii'.format(subject_id),
                     '{}.sulc_MSMAll.32k_fs_LR.dscalar.nii'.format(subject_id),
                     '{}.curvature_MSMAll.32k_fs_LR.dscalar.nii'.format(subject_id),
                     '{}.thickness_MSMAll.32k_fs_LR.dscalar.nii'.format(subject_id)
                    ]
        
        for idx, file in enumerate(file_list):
            file_path = os.path.join(subject_dir, subject_id, 'MNINonLinear', 'fsaverage_LR32k', file)
            if os.path.exists(file_path):
                file_list[idx] = file_path
            else:
                raise RuntimeError
            
        L_sphere_path = file_list[0]
        R_sphere_path = file_list[1]
        L_white_path = file_list[2]
        R_white_path = file_list[3]
        feature_list = file_list[4:]


        points, faces = nibabel.load(L_sphere_path).darrays
        points = np.array(points.data)
        faces = np.array(faces.data)
        faces = np.concatenate((np.ones((faces.shape[0], 1))*3, faces), axis=-1).astype(np.int32)
        L_sphere = pyvista.PolyData(points, faces)

        points, faces = nibabel.load(L_white_path).darrays
        points = np.array(points.data)
        L_sphere['x'] = points[:, 0]
        L_sphere['y'] = points[:, 1]
        L_sphere['z'] = points[:, 2]

        points, faces = nibabel.load(R_sphere_path).darrays
        points = np.array(points.data)
        faces = np.array(faces.data)
        faces = np.concatenate((np.ones((faces.shape[0], 1))*3, faces), axis=-1).astype(np.int32)
        R_sphere = pyvista.PolyData(points, faces)

        points, faces = nibabel.load(R_white_path).darrays
        points = np.array(points.data)
        R_sphere['x'] = points[:, 0]
        R_sphere['y'] = points[:, 1]
        R_sphere['z'] = points[:, 2]

        feature_map_list = []
        for feature_file in feature_list:
            feature = nibabel.load(feature_file).get_data()
            feature = np.array(feature).squeeze()
            feature_map_list.append(feature)
        feature_map_list = np.array(feature_map_list)

        L_feature_map = np.array(feature_map_list[:, hcp_utils.struct.cortex_left])
        R_feature_map = np.array(feature_map_list[:, hcp_utils.struct.cortex_right])

        L_feature = np.zeros((feature_map_list.shape[0], 32492))
        R_feature = np.zeros((feature_map_list.shape[0], 32492))

        L_feature[:, hcp_utils.vertex_info.grayl] = L_feature_map
        R_feature[:, hcp_utils.vertex_info.grayr] = R_feature_map

        L_sphere['sulc'] = L_feature[0, :]
        L_sphere['curv'] = L_feature[1, :]
        L_sphere['thickness'] = L_feature[2, :]

        R_sphere['sulc'] = R_feature[0, :]
        R_sphere['curv'] = R_feature[1, :]
        R_sphere['thickness'] = R_feature[2, :]

        L_save_path = os.path.join(subject_dir, '%s.L.32k_fs_LR.Sphere.vtk'%(subject_id))
        L_sphere.save(L_save_path, binary=False)

        R_save_path = os.path.join(subject_dir, '%s.R.32k_fs_LR.Sphere.vtk'%(subject_id))
        R_sphere.save(R_save_path, binary=False)
    print("True")
    return

def combine_gradient_density():
    """
    Combines gradient density information from separate files into the main surface file.

    This function iterates through a directory of files and combines the gradient density information
    from separate files into the main surface file. It reads the gradient density from the separate
    file and adds it as a new data array to the main surface file. The modified main surface file is
    then saved with a new filename.

    Returns:
        None
    """
    data_dir = "/work/users/j/i/jialec/For_Caochao/HCP_data"
    file_list = os.listdir(data_dir)
    pbar = tqdm(file_list)
    for file in pbar:
        if '.32k_fs_LR.Sphere.vtk' not in file:
            continue
        subject_id = file.split('.')[0]
        file_path = os.path.join(data_dir, file)
        
        if '.L.' in file_path:
            grad_file = os.path.join("/work/users/j/i/jialec/Data/HCP/grad_32k/subjects/%s_3T_rfMRI_REST1_preproc"%subject_id, "gradient_density.lh.Sphere.32k.vtk")
        elif '.R.' in file_path:
            grad_file = os.path.join("/work/users/j/i/jialec/Data/HCP/grad_32k/subjects/%s_3T_rfMRI_REST1_preproc"%subject_id, "gradient_density.rh.Sphere.32k.vtk")
        else:
            raise RuntimeError

        grad_surf = pyvista.read(grad_file)
        grad_density = grad_surf['gradient_density']

        surf = pyvista.read(file_path)
        surf['gradient_density'] = grad_density
        surf.save(file_path.replace('.32k_fs_LR.Sphere.vtk', '.withGrad.32k_fs_LR.Sphere.vtk'), binary=False)
    return

def resample_sphere_32k_164k():
    """
    Resamples the sphere from 32k to 164k resolution for each file in the data directory.
    
    Returns:
        None
    """
    data_dir = "/work/users/j/i/jialec/For_Caochao/HCP_data"
    file_list = os.listdir(data_dir)
    pbar = tqdm(file_list)
    for file in pbar:
        if '.withGrad.32k_fs_LR.Sphere.vtk' not in file:
            continue
        
        file_path = os.path.join(data_dir, file)
        if '.L.' in file_path:
            template = '/proj/ganglilab/FreeSurferLR/fs_L-to-fs_LR_fsaverage.L_LR.spherical_std.164k_fs_L.vtk'
        elif '.R.' in file_path:
            template = '/proj/ganglilab/FreeSurferLR/fs_R-to-fs_LR_fsaverage.R_LR.spherical_std.164k_fs_R.vtk'
        else: 
            raise RuntimeError

        feats = 'sulc+curv+thickness+gradient_density+x+y+z'
        save_path = file_path.replace('.withGrad.32k_fs_LR.Sphere.vtk', '.withGrad.164k_fs_LR.Sphere.vtk')
        cmd = "python ResampleFeatureAndLabel.py --orig_sphe %s --template %s --feats %s --out_name %s" % (file_path, template, feats, save_path)
        print(cmd)
        os.system(cmd)
    return

def transform_sphere_fs_LR_164k_fsaverage():
    """
    Transforms the sphere files from fs_LR_164k_fsaverage to fsaverage.

    This function reads sphere files from the specified directory and transforms them from fs_LR_164k_fsaverage to fsaverage.
    It saves the transformed sphere files with the same name but with the 'fsaverage' suffix.

    Returns:
        None
    """
    data_dir = "/work/users/j/i/jialec/For_Caochao/HCP_data"
    file_list = os.listdir(data_dir)
    pbar = tqdm(file_list)
    for file in pbar:
        if '.withGrad.164k_fs_LR.Sphere.vtk' not in file:
            continue
        
        file_path = os.path.join(data_dir, file)
        if '.L.' in file_path:
            template = pyvista.read('/proj/ganglilab/FreeSurferLR/fsaverage.L.sphere.164k_fs_L.vtk.ply.vtk')
            # template = pyvista.read('/proj/ganglilab/FreeSurferLR/fs_L-to-fs_LR_fsaverage.L_LR.spherical_std.164k_fs_L.vtk')
        elif '.R.' in file_path:
            template = pyvista.read('/proj/ganglilab/FreeSurferLR/fsaverage.R.sphere.164k_fs_R.vtk.ply.vtk')
            # template = pyvista.read('/proj/ganglilab/FreeSurferLR/fs_R-to-fs_LR_fsaverage.R_LR.spherical_std.164k_fs_R.vtk')
        else: 
            raise RuntimeError
        
        points = template.points
        faces = template.faces
        faces = np.array(faces).reshape((-1, 4))
        sphere = pyvista.PolyData(points, faces)

        surf = pyvista.read(file_path)
        feats = ['sulc', 'curv', 'thickness', 'gradient_density', 'x', 'y','z']
        for feat in feats:
            sphere[feat] = surf[feat]

        sphere.save(file_path.replace('.withGrad.164k_fs_LR.Sphere.vtk', '.withGrad.164k_fsaverage.Sphere.vtk'))
    print('True')
    return

def create_inner_vtk():
    """
    Create inner VTK files from the given data directory.

    This function reads VTK files from the specified data directory, processes them, and saves the inner VTK files.
    Only files with the suffix '.withGrad.164k_fsaverage.flip.rescale.Sphere.vtk' are processed.

    Returns:
        None
    """
    data_dir = "/work/users/j/i/jialec/For_Caochao/HCP_data"
    file_list = os.listdir(data_dir)
    pbar = tqdm(file_list)
    for file in pbar:
        if '.withGrad.164k_fsaverage.flip.rescale.Sphere.vtk' not in file:
            continue
        file_path = os.path.join(data_dir, file)

        sphere = pyvista.read(file_path)
        faces = np.array(sphere.faces)
        faces = np.array(faces).reshape((-1, 4))

        x = sphere['x']
        y = sphere['y']
        z = sphere['z']

        points = np.array([x, y, z]).transpose((1, 0))

        inner = pyvista.PolyData(points, faces)
        inner['sulc'] = sphere['sulc']
        inner['curv'] = sphere['curv']
        inner['thickness'] = sphere['thickness']
        inner['gradient_density'] = sphere['gradient_density']
        inner.save(file_path.replace('.withGrad.164k_fsaverage.flip.rescale.Sphere.vtk', '.withGrad.164k_fsaverage.flip.rescale.Inner.vtk'), binary=False)
    return

def create_morph_data():
    """
    Creates morphological data for each subject in the given data directory.

    Returns:
        None
    """
    data_dir = "/work/users/j/i/jialec/For_Caochao/HCP_data"
    file_list = os.listdir(data_dir)
    pbar = tqdm(file_list)
    for file in pbar:
        if '.withGrad.164k_fsaverage.Inner.vtk' not in file:
            continue
        subject_id = file.split('.')[0]
        if '.L.' in file:
            hemi = 'lh'
        elif '.R.' in file:
            hemi = 'rh'

        file_path = os.path.join(data_dir, file)
        surf = pyvista.read(file_path)
        sulc = surf['sulc']
        curv = surf['curv']
        thickness = surf['thickness']

        fio.write_morph_data(os.path.join(data_dir, "%s.%s.sulc"%(subject_id, hemi)), sulc, fnum=327680)
        fio.write_morph_data(os.path.join(data_dir, "%s.%s.curv"%(subject_id, hemi)), curv, fnum=327680)
        fio.write_morph_data(os.path.join(data_dir, "%s.%s.thickness"%(subject_id, hemi)), thickness, fnum=327680)
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

        fio.write_morph_data(os.path.join(data_dir, "%s.%s.flip.sulc"%(subject_id, hemi)), sulc, fnum=327680)
        fio.write_morph_data(os.path.join(data_dir, "%s.%s.flip.curv"%(subject_id, hemi)), curv, fnum=327680)
        sphere.save(file_path.replace('.withGrad.164k_fsaverage.Sphere.vtk', '.withGrad.164k_fsaverage.flip.Sphere.vtk'), binary=False)
    return

def rescale_feature():
    """
    Rescales the 'sulc' and 'curv' features of each file in the specified data directory.

    Returns:
        None
    """
    data_dir = "/work/users/j/i/jialec/For_Caochao/HCP_data"
    file_list = os.listdir(data_dir)
    pbar = tqdm(file_list)
    for file in pbar:
        if '.withGrad.164k_fsaverage.flip.Sphere.vtk' not in file:
            continue
        subject_id = file.split('.')[0]
        if '.L.' in file:
            hemi = 'lh'
        elif '.R.' in file:
            hemi = 'rh'

        file_path = os.path.join(data_dir, file)

        sphere = pyvista.read(file_path)
        sulc = sphere['sulc']
        curv = sphere['curv']
        
        sulc *= (1.78 + 1.88) / (np.max(sulc) - np.min(sulc))
        sulc -= np.max(sulc) - 1.78

        curv *= (0.54 + 0.67) / (np.max(curv) - np.min(curv))
        curv -= np.max(curv) - 0.54

        sphere['sulc'] = sulc
        sphere['curv'] = curv

        fio.write_morph_data(os.path.join(data_dir, "%s.%s.flip.rescale.sulc"%(subject_id, hemi)), sulc, fnum=327680)
        fio.write_morph_data(os.path.join(data_dir, "%s.%s.flip.rescale.curv"%(subject_id, hemi)), curv, fnum=327680)
        sphere.save(file_path.replace('.withGrad.164k_fsaverage.flip.Sphere.vtk', '.withGrad.164k_fsaverage.flip.rescale.Sphere.vtk'), binary=False)
    return

if __name__ == "__main__":
    # delete_this()
    # create_vtk()
    # combine_gradient_density()
    # resample_sphere_32k_164k()
    # transform_sphere_fs_LR_164k_fsaverage()
    create_inner_vtk()
    # create_morph_data()
    # flip_feature()
    # rescale_feature()
    print("True")

    # surf_L = pyvista.read("/media/jialec/My Book/DATA/For_Caochao/Raw_data/100206_3T_Structural_preproc/100206.L.164k_fs_LR.Sphere.vtk")
    # surf_R = pyvista.read("/media/jialec/My Book/DATA/For_Caochao/Raw_data/100206_3T_Structural_preproc/100206.R.164k_fs_LR.Sphere.vtk")

    # point_L = surf_L.points
    # point_R = surf_R.points

    # surf = pyvista.read("/home/jialec/Code/For_Caochao/100206.R.withGrad.164k_fsaverage.Inner.vtk")
    # faces = surf.faces
    # faces = np.array(faces).reshape((-1, 4))

    # lh_thickness = fio.read_morph_data('./20240608_prepare_hcp_fmri/100206.lh.thickness')
    # lh_sulc = fio.read_morph_data('./20240608_prepare_hcp_fmri/100206.lh.sulc')
    # print("True")

