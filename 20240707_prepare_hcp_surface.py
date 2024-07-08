import os
import pyvista
import nibabel
import hcp_utils
import numpy as np
from tqdm import tqdm
from nibabel.freesurfer import io as fio

def create_vtk():
    save_dir = "/work/users/j/i/jialec/For_Caochao/HCP_inner_fromFenqiang_20240707"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    data_dir = "/work/users/l/a/laifama/Fengqiang/LifeSpanDataset/HCP/FSProcessedResults_withOrigRes"
    subject_list = os.listdir(data_dir)
    pbar = tqdm(subject_list)
    for subject in pbar:
        subject_dir = os.path.join(data_dir, subject)
        if 'fsaverage' in subject:
            continue
        if '.out' in subject:
            continue
        subject_id = subject.split('_')[0]
        save_subejct = os.path.join(save_dir, subject_id)
        if not os.path.exists(save_subejct):
            os.mkdir(save_subejct)

        L_inner_path = os.path.join(subject_dir, "%s.lh.InnerSurf.vtk"%subject)
        L_sphere_path = os.path.join(subject_dir, "%s.lh.SpheSurf.RegByFS.vtk"%subject)

        R_inner_path = os.path.join(subject_dir, "%s.rh.InnerSurf.vtk"%subject)
        R_sphere_path = os.path.join(subject_dir, "%s.rh.SpheSurf.RegByFS.vtk"%subject)

        L_save_path = os.path.join(save_subejct, "%s.lh.SpheSurf.RegByFS.vtk"%subject)
        if not os.path.exists(L_save_path):
            if os.path.exists(L_inner_path) and os.path.exists(L_sphere_path):
                surf = pyvista.read(L_sphere_path)
                points = np.array(surf.points)
                faces = np.array(surf.faces).reshape((-1, 4))
                # faces = np.concatenate((np.ones((faces.shape[0], 1))*3, faces), axis=-1).astype(np.int32)
                L_sphere = pyvista.PolyData(points, faces)

                inner_surf = pyvista.read(L_inner_path)
                points = np.array(inner_surf.points)
                L_sphere['x'] = points[:, 0]
                L_sphere['y'] = points[:, 1]
                L_sphere['z'] = points[:, 2]

                L_sphere.save(L_save_path, binary=False)

        R_save_path = os.path.join(save_subejct, "%s.rh.SpheSurf.RegByFS.vtk"%subject)
        if not os.path.exists(R_save_path):
            if os.path.exists(R_inner_path) and os.path.exists(R_sphere_path):

                surf = pyvista.read(R_sphere_path)
                points = np.array(surf.points)
                faces = np.array(surf.faces).reshape((-1, 4))
                # faces = np.concatenate((np.ones((faces.shape[0], 1))*3, faces), axis=-1).astype(np.int32)
                R_sphere = pyvista.PolyData(points, faces)

                inner_surf = pyvista.read(R_inner_path)
                points = np.array(inner_surf.points)
                R_sphere['x'] = points[:, 0]
                R_sphere['y'] = points[:, 1]
                R_sphere['z'] = points[:, 2]

                R_sphere.save(R_save_path, binary=False)
        # break
    print("True")
    return


def resample_sphere():
    data_dir = "/work/users/j/i/jialec/For_Caochao/HCP_inner_fromFenqiang_20240707"
    subject_list = os.listdir(data_dir)
    pbar = tqdm(subject_list)
    for subject in pbar:
        subject_dir = os.path.join(data_dir, subject)
        lh_file = os.path.join(subject_dir, "%s.lh.SpheSurf.RegByFS.vtk"%subject)
        if os.path.exists(lh_file):
            template = "/work/users/j/i/jialec/code/LifeSpanAtlas/scripts/atlas/SparseAtlas_72_lh.SphereSurf.vtk"

            feats = 'x+y+z'
            save_path = lh_file.replace('.vtk', '.Resp163842.vtk')
            # cmd = "python ResampleFeatureAndLabel.py --orig_sphe %s --template %s --feats %s --out_name %s" % (lh_file, template, feats, save_path)
            cmd = "sbatch -p general -N 1 -n 1 --mem=16g -t 01:00:00 --wrap=\"python ResampleFeatureAndLabel.py --orig_sphe %s --template %s --feats %s --out_name %s\"" % (lh_file, template, feats, save_path)
            print(cmd)
            os.system(cmd)

        rh_file = os.path.join(subject_dir, "%s.rh.SpheSurf.RegByFS.vtk"%subject)
        if os.path.exists(rh_file):
            template = "/work/users/j/i/jialec/code/LifeSpanAtlas/scripts/atlas/SparseAtlas_72_rh.SphereSurf.vtk"
        
            feats = 'x+y+z'
            save_path = rh_file.replace('.vtk', '.Resp163842.vtk')
            # cmd = "python ResampleFeatureAndLabel.py --orig_sphe %s --template %s --feats %s --out_name %s" % (rh_file, template, feats, save_path)
            cmd = "sbatch -p general -N 1 -n 1 --mem=16g -t 01:00:00 --wrap=\"python ResampleFeatureAndLabel.py --orig_sphe %s --template %s --feats %s --out_name %s\"" % (rh_file, template, feats, save_path)
            print(cmd)
            os.system(cmd)
        # break
    return


def create_inner_vtk():
    data_dir = "/work/users/j/i/jialec/For_Caochao/HCP_inner_fromFenqiang_20240707"
    subject_list = os.listdir(data_dir)
    pbar = tqdm(subject_list)
    for subject in pbar:
        # if '.withGrad.164k_fsaverage.flip.Sphere.vtk' not in file:
        subject_dir = os.path.join(data_dir, subject)

        lh_file = os.path.join(subject_dir, "%s.lh.SpheSurf.RegByFS.Resp163842.vtk"%subject)

        if os.path.exists(lh_file):
            sphere = pyvista.read(lh_file)
            faces = np.array(sphere.faces)
            faces = np.array(faces).reshape((-1, 4))
            # faces = np.concatenate((np.ones((faces.shape[0], 1))*3, faces), axis=-1).astype(np.int32)

            x = sphere['x']
            y = sphere['y']
            z = sphere['z']

            points = np.array([x, y, z]).transpose((1, 0))

            inner = pyvista.PolyData(points, faces)
            inner.save(lh_file.replace('.SpheSurf.', '.InnerSurf.'), binary=False)

        rh_file = os.path.join(subject_dir, "%s.rh.SpheSurf.RegByFS.Resp163842.vtk"%subject)

        if os.path.exists(rh_file):
            sphere = pyvista.read(rh_file)
            faces = np.array(sphere.faces)
            faces = np.array(faces).reshape((-1, 4))
            # faces = np.concatenate((np.ones((faces.shape[0], 1))*3, faces), axis=-1).astype(np.int32)

            x = sphere['x']
            y = sphere['y']
            z = sphere['z']

            points = np.array([x, y, z]).transpose((1, 0))

            inner = pyvista.PolyData(points, faces)
            inner.save(rh_file.replace('.SpheSurf.', '.InnerSurf.'), binary=False)
        # break
    return


if __name__ == "__main__":
    # create_vtk()
    # resample_sphere()
    create_inner_vtk()
    
