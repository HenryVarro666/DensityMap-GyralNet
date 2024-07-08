import os
import tqdm

def gyralnet_generate():
    data_dir = "/work/users/c/h/chaocao/HCP_structure_FromFenqiang"
    subject_list = os.listdir(data_dir)
    pbar = tqdm(subject_list)
    for subject in pbar:
        output_path = os.path.join(data_dir, subject, subject +  "_GDM_net")
        subject_recon_dir = os.path.join(data_dir, subject, subject + "_recon", "surf")
        for hemi in ["lh", "rh"]:
            sphere_file = os.path.join(subject_recon_dir, "%s.%s.SpheSurf.RegByFS.Resp163842.vtk"%(subject, hemi))
            inner_file = os.path.join(subject_recon_dir, "%s.%s.InnerSurf.RegByFS.Resp163842.vtk"%(subject, hemi))
            grad_file = os.path.join(subject_recon_dir, "%s.grad"%hemi)
            rescale_grad_file = os.path.join(subject_recon_dir, "%s.rescale.grad"%hemi)
            if os.path.exists(sphere_file, inner_file):
                # cmd = "python ResampleFeatureAndLabel.py --orig_sphe %s --template %s --feats %s --out_name %s" % (lh_file, template, feats, save_path)
                cmd = "sbatch -p general -N 1 -n 1 --mem=16g -t 01:00:00 --wrap=\"python ResampleFeatureAndLabel.py --orig_sphe %s --template %s --feats %s --out_name %s\"" % (lh_file, template, feats, save_path)
                print(cmd)
                os.system(cmd)
    return

if __name__ == "__main__":
    gyralnet_generate()
    