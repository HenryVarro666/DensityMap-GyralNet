'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-06-30 23:16:01
FilePath: /DensityMap-GyralNet/GDM.py
'''
import vtk
import nibabel.freesurfer.io as io
from collections import defaultdict # 用于创建带有默认值的字典
import pdb
import numpy as np
import os
import shutil
import time
import networkx as nx
import argparse

from utils.featured_sphere import Writer
from utils.points import Points
from utils.sulc_data import SulcData
from utils.vtkutils import Draw, Read
from utils.gyri_data import GyralData
from utils.tree import Tree  
from utils.connect import Connection  

def main(args):
    # root = args.root_dir
    # subject_index_start = args.subject_list_start_id
    # # print(subject_index_start)
    # subject_index_end = args.subject_list_end_id
    # # subjects_list = [str(subject) for subject in os.listdir(root) if not subject.startswith('.')]
    # # subjects_list.sort()
    # # current_subject_list = subjects_list[subject_index_start:subject_index_end]
    # current_subject_list = [subject_index_start]
    # print(current_subject_list)
    # sphere_list = args.sphere_list


    root = "."
    sphere_list = args.sphere_list
    current_subject_list = [100206]
    for subject in current_subject_list:
        subject = str(subject)
        print(subject)

        out_dir = root + '/' + subject + '/' + 'GDM_tmp'

        if not os.path.exists(out_dir):
            # shutil.rmtree(out_dir)
            os.mkdir(out_dir)
            print("Output_dir is: " + out_dir)

        # sphere_list = ['lh', 'rh']
        for sphere in sphere_list:
            result_file = f'{sphere}_surf_3hinge_vertex.vtk'
            result_file_path = os.path.join(out_dir, result_file)
            print(result_file_path)

            if os.path.exists(result_file_path):
                continue

            recon_folder = str(subject + '_recon')

            sphere_file = root + '/' + subject + '/' + recon_folder + '/' + 'surf' + '/' +sphere + args.sphere_file
            print(sphere_file)
            surf_file = root + '/' + subject +  '/' + recon_folder + '/' + 'surf' + '/' +sphere + args.surf_file
            print(surf_file)
            curv_file = root + '/' + subject + '/' + recon_folder + '/' + 'surf' + '/' +sphere + args.curv_file
            print(curv_file)
            sulc_file = root + '/' + subject + '/' + recon_folder + '/' + 'surf' + '/' +sphere + args.sulc_file
            print(sulc_file)
            thickness_file = root +'/' + subject + '/' + recon_folder + '/' + 'surf' + '/' + sphere + args.thickness_file
            print(thickness_file)


            output_prefix = out_dir + '/' + sphere
            feature_file_dict = {'sulc': sulc_file, 'curv': curv_file, 'thickness': thickness_file}
            print(feature_file_dict)


            sphere_polydata = Read.read_vtk_file(sphere_file)
            surf_polydata = Read.read_vtk_file(surf_file)

            ## Binary
            # _sphere_features.vtk
            # _surf_features.vtk
            Writer.write_featured_sphere(sphere_polydata, feature_file_dict, output_prefix + '_sphere_features.vtk')
            Writer.write_featured_sphere(surf_polydata, feature_file_dict, output_prefix + '_surf_features.vtk')
            
            print('create points connection dict:\t' + time.asctime(time.localtime(time.time())))
            # 每个顶点与其相连的其他顶点的信息
            point_neighbor_points_dict = Points.get_connect_points(sphere_polydata)

            print('initialize sulc data:\t' + time.asctime(time.localtime(time.time())))
            # 初始化sulc数据
            updated_sulc_data, original_sulc_data, curv_data_delete_thicknessZero = SulcData.initialize_sulc_data(sphere_polydata,
                                                                                                         surf_polydata,
                                                                                                         feature_file_dict,
                                                                                                         point_neighbor_points_dict,
                                                                                                         args.inner_sulci_curv_thres,
                                                                                                         args.inner_sulci_round_thres,
                                                                                                         args.outer_gyri_curv_thres,
                                                                                                         args.inner_sulci_neighbor_curv_thres,
                                                                                                         output_prefix)

            print('calculate patchsize of gyri part:\t' + time.asctime(time.localtime(time.time())))
            # 计算gyri部分的patchsize
            point_patchSize_dict_updated = GyralData.find_the_patchSize_of_gyri_point(updated_sulc_data, point_neighbor_points_dict)


            print('draw patchSize colorful sphere:\t' + time.asctime(time.localtime(time.time())))
            Draw.draw_patchSize_colorful_file(sphere_polydata, surf_polydata, point_patchSize_dict_updated, output_prefix, 'updated')
            
            thin_sulc_data = Writer.write_thin_gyri_on_sphere_point_marginal_sulc_curv(sphere_polydata, surf_polydata,
                                                                                point_neighbor_points_dict,
                                                                                updated_sulc_data,
                                                                                curv_data_delete_thicknessZero,
                                                                                args.expend_curv_step_size,
                                                                                output_prefix)
            print('create thin gyri connection dict \t' + time.asctime(time.localtime(time.time())))

            # pdb.set_trace()

            point_connect_points_dict_thin_gyri_parts = Points.get_connect_points_gyri_part(sphere_polydata, thin_sulc_data)
            
            connected_lines_list, father_dict = Tree.create_tree(sphere_polydata, surf_polydata, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, thin_sulc_data, output_prefix)
            
            Tree.find_skelenton(sphere_polydata, surf_polydata, point_patchSize_dict_updated, curv_data_delete_thicknessZero,
                           original_sulc_data, thin_sulc_data, point_neighbor_points_dict, point_connect_points_dict_thin_gyri_parts,
                           connected_lines_list, father_dict, args.length_thres_of_long_gyri, args.neighbor_missing_path_smallest_step,
                           args.flat_threshold_for_convex_gyri, args.nearest_skeleton_num, args.island_gyri_length_thres, output_prefix)

            skeleton_polydata = Read.read_vtk_file(output_prefix + '_sphere_skelenton_connect_break_allpoints.vtk')
            Connection.find_skelenton_missing(sphere_polydata, surf_polydata, skeleton_polydata, curv_data_delete_thicknessZero,
                                   original_sulc_data,
                                   args.length_thres_of_long_gyri, args.neighbor_missing_path_smallest_step,
                                   args.flat_threshold_for_convex_gyri, args.nearest_skeleton_num, args.island_gyri_length_thres, output_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GyralNet creation by expending algorithm")
    # parser.add_argument('-root_dir', '--root_dir', type=str, default='./', help='root for input data')

    # parser.add_argument('-subject_list_start_id', '--subject_list_start_id', type=int, default=0, help='subjects list start and end ids')
    # parser.add_argument('-subject_list_end_id', '--subject_list_end_id', type=int, default=-1, help='subjects list start and end ids')

    # parser.add_argument('-input_dir', '--input_dir', type=str, default='100206_recon', help='input dir within each subject')
    # parser.add_argument('-out_dir', '--out_dir', type=str, default='gyralnet_island_tmp', help='out dir within each subject')
    parser.add_argument('-sphere_list', '--sphere_list', type=list, default=['lh', 'rh'], help='spheres')
    parser.add_argument('-sphere_file', '--sphere_file', type=str, default='.withGrad.32k_fs_LR.Sphere.vtk', help='sphere_file name')

    parser.add_argument('-surf_file', '--surf_file', type=str, default='.withGrad.32k_fs_LR.Inner.vtk', help='surf_file name')
    parser.add_argument('-curv_file', '--curv_file', type=str, default='.flip.curv', help='curv_file name')
    parser.add_argument('-sulc_file', '--sulc_file', type=str, default='.flip.sulc', help='sulc_file name')
    parser.add_argument('-thickness_file', '--thickness_file', type=str, default='.thickness', help='thickness_file name')

    # inner_sulci_curv_thres (float): the curv threshold to identify sulci in the gyri part
    parser.add_argument('-inner_sulci_curv_thres', '--inner_sulci_curv_thres', type=float, default=0.06, help='the curv threshold to identify sulci in the gyri part')
    parser.add_argument('-inner_sulci_neighbor_curv_thres', '--inner_sulci_neighbor_curv_thres', type=float, default=0.006, help='the curv threshold to change the neighbor to sulci')
    parser.add_argument('-inner_sulci_round_thres', '--inner_sulci_round_thres', type=int, default=6, help='the patch size threshold to identify sulci in the gyri part')
    parser.add_argument('-outer_gyri_curv_thres', '--outer_gyri_curv_thres', type=int, default=-0.03, help='the curv threshold to identify gyri in the sulci part')
    parser.add_argument('-expend_curv_step_size', '--expend_curv_step_size', type=float, default=0.02, help='during expending, the edges with small curv will be erosen first')
    parser.add_argument('-length_thres_of_long_gyri', '--length_thres_of_long_gyri', type=int, default=6, help='length threshold to select missing long gyri')
    parser.add_argument('-neighbor_missing_path_smallest_step', '--neighbor_missing_path_smallest_step', type=int, default=15, help='length threshold to select missing long gyri')
    parser.add_argument('-flat_threshold_for_convex_gyri', '--flat_threshold_for_convex_gyri', type=int, default=4, help='to evaluate if the long path represent a convex gyri')
    parser.add_argument('-nearest_skeleton_num', '--nearest_skeleton_num', type=int, default=6, help='the number of nearest skeleton points to each marginal convex point')
    parser.add_argument('-island_gyri_length_thres', '--island_gyri_length_thres', type=int, default=20, help='the lower bound the the length of island gyri to be connected')

    args = parser.parse_args()
    print(args)
    main(args)