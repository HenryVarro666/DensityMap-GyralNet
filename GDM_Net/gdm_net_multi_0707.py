import nibabel.freesurfer.io as io
import numpy as np
import pyvista
import vtk
import os
from collections import defaultdict
import time
import shutil

from utils.data_io import read_vtk_file, clear_output_folder, check_output_folder
from utils.draw import write_featured_sphere_from_variable, write_featured_sphere_from_variable_single
# from utils.draw import featured_sphere
from utils.draw import get_connect_points, draw_patchSize_colorful_file
from utils.augmentation import delete_isolated_point, find_marginal_point
from utils.tree import get_connect_points_gyri_part, create_tree
from utils.SkeletonProcessing import Find_skelenton


def initialize_grad_data(orig_sphere_polydata, orig_surf_polydata, feature_file_dict_grad, point_neighbor_points_dict,
                        output_prefix, sphere, grad_threshold):
    point_num = orig_sphere_polydata.GetNumberOfPoints()

    grad_data_raw = io.read_morph_data(feature_file_dict_grad['gradient_density'])
    
    flip_grad_data_raw = -grad_data_raw

    grad_data_rescale = io.read_morph_data(feature_file_dict_grad['recale_gradient_density'])

    flip_grad_data_rescale = -grad_data_rescale

    grad_data_binary = np.where(flip_grad_data_raw <= grad_threshold, -1, 1)

    grad_data_updated = delete_isolated_point(point_num, point_neighbor_points_dict, grad_data_binary)

    print('Draw updated gradient_density colorful sphere:\t' + time.asctime(time.localtime(time.time())))
    feature_name_variable_dict = {'grad_data_raw': grad_data_raw,
                                  'flip_grad_data_raw': flip_grad_data_raw,
                                  'grad_data_rescale': grad_data_rescale,
                                  'flip_grad_data_rescale': flip_grad_data_rescale,
                                  'grad_data_binary': grad_data_binary,
                                  'grad_data_updated': grad_data_updated}
    sphere_output = os.path.join(output_prefix, "%s_sphere_feature_updated_rescale_%s.vtk"%(sphere,grad_threshold))
    surf_output = os.path.join(output_prefix, "%s_surf_featured_updated_rescale_%s.vtk"%(sphere,grad_threshold))
    write_featured_sphere_from_variable(orig_sphere_polydata, feature_name_variable_dict, sphere_output)
    write_featured_sphere_from_variable(orig_surf_polydata, feature_name_variable_dict, surf_output)
    
    return grad_data_raw, flip_grad_data_raw, grad_data_rescale, flip_grad_data_rescale, grad_data_binary, grad_data_updated

def find_the_patchSize_of_gyri_point(grad_data_updated, point_neighbor_points_dict):
    point_patchSize_dict = dict()
    for point in point_neighbor_points_dict.keys():
        if grad_data_updated[point] > 0:
            point_patchSize_dict[point] = -1
        else:
            inner_points_list = list()
            outer_points_list = list()
            outer_points_list.append(point)
            round_num = 0
            while np.sum(grad_data_updated[outer_points_list] < 0) == len(outer_points_list):
                round_num += 1
                inner_points_list = inner_points_list + outer_points_list
                next_outer_candidate = list()
                for outer_point in outer_points_list:
                    neighbor_points = [neighbor_point for neighbor_point in point_neighbor_points_dict[outer_point] if neighbor_point not in inner_points_list]
                    next_outer_candidate = next_outer_candidate + neighbor_points

                outer_points_list = list(set(next_outer_candidate))
            point_patchSize_dict[point] = round_num
    return point_patchSize_dict

def write_thin_gyri_on_sphere_point_marginal_sulc_curv(orig_sphere_polydata, 
                                                       orig_surf_polydata, 
                                                       point_neighbor_points_dict, 
                                                       grad_data_updated, 
                                                       flip_grad_data_rescale,  
                                                       output_prefix,
                                                       sphere,
                                                       threshold,
                                                       expend_step_size):
    
    print('================= Begin erosion:\t' + time.asctime(time.localtime(time.time())) + '=========================')
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    cell_num = orig_sphere_polydata.GetNumberOfCells()
    CellArray = orig_sphere_polydata.GetPolys()
    Polygons = CellArray.GetData()

    print('create triangles connection:\t' + time.asctime(time.localtime(time.time())))
    triangle_collection = list()
    for cell in range(cell_num):
        triangle = set([Polygons.GetValue(j) for j in range(cell * 4 + 1, cell * 4 + 4)])
        triangle_collection.append(triangle)

    print('Find initial marginal points:\t' + time.asctime(time.localtime(time.time())))
    marginal_points, marginal_points_gyri, marginal_points_sulc = find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, grad_data_updated)
    # pdb.set_trace()


    print('Begin expending:\t' + time.asctime(time.localtime(time.time())))
    round = 0
 
    min_grad = flip_grad_data_rescale.min()
    current_grad = threshold
    grad_step = expend_step_size

    while current_grad >= min_grad:
        current_grad = current_grad - grad_step
        finish = 1
        print('\n================ current grad:', current_grad, '====================')
        while finish > 0:
            round += 1
            print('round', str(round), ':\t', time.asctime(time.localtime(time.time())))
            finish = 0
            redlize_points = list()
            marginal_points_candidate = list()
            for point in marginal_points_sulc:
                grad = grad_data_updated[point]
                if grad < 0:
                    print('updated_grad_data grad value error!')
                    exit()
                else:
                    # neighbor_points = point_neighbor_points_dict[point]
                    neighbor_points = [neighbor for neighbor in point_neighbor_points_dict[point]]
                    for neighbor_point in neighbor_points:
                        if grad_data_updated[neighbor_point] < 0 and flip_grad_data_rescale[neighbor_point] >= current_grad:
                            second_neighbors = point_neighbor_points_dict[neighbor_point]
                            marginal_gyri_num = 0
                            marginal_gyri_points = list()
                            for second_neighbor in second_neighbors:
                                if grad_data_updated[second_neighbor] < 0 and second_neighbor in marginal_points_gyri:
                                    marginal_gyri_num += 1
                                    marginal_gyri_points.append(second_neighbor)
                            if (marginal_gyri_num < 2) or (marginal_gyri_num == 2 and (np.sum(grad_data_updated[second_neighbors] < 0) > 2 or set(marginal_gyri_points + [neighbor_point]) in triangle_collection)):
                                redlize_points.append(neighbor_point)
                                marginal_points_candidate = list(set(marginal_points_candidate + second_neighbors))
                                finish = 1
            grad_data_updated[redlize_points] = -1
            marginal_points, marginal_points_gyri, marginal_points_sulc = find_marginal_point(list(set(marginal_points + marginal_points_candidate)), point_neighbor_points_dict, grad_data_updated)
            # pdb.set_trace()
            if round % 500 == 0:
                output = os.path.join(output_prefix, "%s_sphere_round(%d)_marginal_point.vtk"%(sphere, round))
                write_featured_sphere_from_variable(orig_sphere_polydata, {'grad_updated': grad_data_updated}, output)

    print('draw final thin path on sphere:\t' + time.asctime(time.localtime(time.time())))

    sphere_output = os.path.join(output_prefix, "%s_sphere_thin.vtk"%(sphere))
    surf_output = os.path.join(output_prefix, "%s_surf_thin.vtk"%(sphere))

    write_featured_sphere_from_variable(orig_sphere_polydata, {'grad_updated': grad_data_updated}, sphere_output)
    write_featured_sphere_from_variable(orig_surf_polydata, {'grad_updated': grad_data_updated}, surf_output)
    return grad_data_updated


def __main__():
    grad_threshold = -0.5
    expend_step_size = 0.01

    # root = "/home/lab/Documents/DensityMap-GyralNet/32k_3subjects"

    # directories = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d.isdigit()]
    # for subject_id in directories:
    #     subject_id = str(subject_folder)
    #     subject_folder = os.path.join(root, subject_id, subject_id + "_recon", "surf")
    #     print("Subject folder: %s"%(subject_folder))

    root = os.getcwd()
    subject_id = str(os.path.basename(root))
    print(subject_id)
    subject_folder = os.path.join(root, subject_id + "_recon", "surf")
    output_folder_name = subject_id +"_gdm_net_" + str(grad_threshold)
    output_folder = os.path.join(root, output_folder_name)

    # clear_output_folder(output_folder)
    check_output_folder(output_folder)

    # sphere_list = ["lh", "rh"]
    sphere_list = ["lh"]

    for sphere in sphere_list:
        sphere_file = os.path.join(subject_folder, "%s.withGrad.32k_fs_LR.Sphere.vtk"%(sphere))
        surf_file = os.path.join(subject_folder, "%s.withGrad.32k_fs_LR.Inner.vtk"%(sphere))

        sphere_polydata = read_vtk_file(sphere_file)
        surf_polydata = read_vtk_file(surf_file)

        point_neighbor_points_dict = get_connect_points(sphere_polydata)

        point_num = sphere_polydata.GetNumberOfPoints()

        print('Rescale gradient density:\t' + time.asctime(time.localtime(time.time())))
        grad_file = os.path.join(subject_folder, "%s.grad"%(sphere))
        rescale_grad_file = os.path.join(subject_folder, "%s.rescale.grad"%(sphere))


        feature_file_dict_grad =  {'recale_gradient_density': rescale_grad_file,
                                'gradient_density': grad_file}

        # featured_sphere(sphere_polydata, feature_file_dict_grad, os.path.join(output_folder, "%s_sphere_Test.vtk"%(sphere)))
        # featured_sphere(surf_polydata, feature_file_dict_grad, os.path.join(output_folder, "%s_surf_Test.vtk"%(sphere)))


        print('Initialize grad data:\t' + time.asctime(time.localtime(time.time())))

        # Binary
        grad_data_raw, flip_grad_data_raw, grad_data_rescale, flip_grad_data_rescale, grad_data_binary, grad_data_updated= initialize_grad_data(sphere_polydata, 
                            surf_polydata, 
                            feature_file_dict_grad, 
                            point_neighbor_points_dict, 
                            output_folder, 
                            sphere, 
                            grad_threshold) 
        
        print('Calculate patchsize for each gyri point:\t' + time.asctime(time.localtime(time.time())))
        point_patchSize_dict_updated = find_the_patchSize_of_gyri_point(grad_data_updated, point_neighbor_points_dict)
        
        print('Draw patchSize colorful sphere:\t' + time.asctime(time.localtime(time.time())))
        draw_patchSize_colorful_file(sphere_polydata, surf_polydata, point_patchSize_dict_updated, output_folder, sphere)

        thin_sulc_data = write_thin_gyri_on_sphere_point_marginal_sulc_curv(sphere_polydata, 
                                                                            surf_polydata,
                                                                            point_neighbor_points_dict,
                                                                            grad_data_updated,
                                                                            grad_data_rescale,
                                                                            output_folder,
                                                                            sphere,
                                                                            grad_threshold,
                                                                            expend_step_size)
        
        print('create thin gyri connection dict \t' + time.asctime(time.localtime(time.time())))
        
        point_connect_points_dict_thin_gyri_parts = get_connect_points_gyri_part(sphere_polydata, thin_sulc_data)
        connected_lines_list, father_dict = create_tree(sphere_polydata, surf_polydata, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, thin_sulc_data, output_folder, sphere)

        length_thres_of_long_gyri=6
        neighbor_missing_path_smallest_step=15
        flat_threshold_for_convex_gyri=4
        nearest_skeleton_num=6
        island_gyri_length_thres=20


        Find_skelenton.find_skelenton(sphere_polydata, surf_polydata, point_patchSize_dict_updated, grad_data_updated,
                        flip_grad_data_rescale, thin_sulc_data, point_neighbor_points_dict, point_connect_points_dict_thin_gyri_parts,
                        connected_lines_list, father_dict, 
                        length_thres_of_long_gyri, 
                        neighbor_missing_path_smallest_step,
                        flat_threshold_for_convex_gyri, 
                        nearest_skeleton_num, 
                        island_gyri_length_thres,
                        output_folder, 
                        sphere)

        sphere_skelenton_connect_break_allpoints_path = os.path.join(output_folder, "%s_sphere_skelenton_connect_break_allpoints.vtk"%(sphere))
        skeleton_polydata = read_vtk_file(sphere_skelenton_connect_break_allpoints_path)
        Find_skelenton.find_skelenton_missing(sphere_polydata, surf_polydata, skeleton_polydata, 
                                grad_data_updated,
                                flip_grad_data_rescale,
                                length_thres_of_long_gyri, 
                                neighbor_missing_path_smallest_step,
                                flat_threshold_for_convex_gyri,
                                nearest_skeleton_num, 
                                island_gyri_length_thres,
                                output_folder, 
                                sphere)

if __name__ == "__main__":
  __main__()