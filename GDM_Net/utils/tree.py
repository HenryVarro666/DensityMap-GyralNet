import nibabel.freesurfer.io as io
import numpy as np
import pyvista
import vtk
import os
from collections import defaultdict
import time
import shutil

from utils.data_io import read_vtk_file, clear_output_folder
from utils.draw import write_featured_sphere_from_variable, write_featured_sphere_from_variable_single
from utils.draw import get_connect_points, draw_patchSize_colorful_file
from utils.augmentation import delete_isolated_point
from utils.draw import write_skelenton_by_connectionPair

def get_connect_points_gyri_part(surf_polydata, sulc_data):
    point_connect_points_dict_gyri_parts = defaultdict(list)
    cell_num = surf_polydata.GetNumberOfCells()
    CellArray = surf_polydata.GetPolys()
    Polygons = CellArray.GetData()
    for i in range(0, cell_num):
        triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
        if sulc_data[triangle[0]] < 0 and sulc_data[triangle[1]] < 0:
            if triangle[1] not in point_connect_points_dict_gyri_parts[triangle[0]]:
                point_connect_points_dict_gyri_parts[triangle[0]].append(triangle[1])
            if triangle[0] not in point_connect_points_dict_gyri_parts[triangle[1]]:
                point_connect_points_dict_gyri_parts[triangle[1]].append(triangle[0])

        if sulc_data[triangle[0]] < 0 and sulc_data[triangle[2]] < 0:
            if triangle[2] not in point_connect_points_dict_gyri_parts[triangle[0]]:
                point_connect_points_dict_gyri_parts[triangle[0]].append(triangle[2])
            if triangle[0] not in point_connect_points_dict_gyri_parts[triangle[2]]:
                point_connect_points_dict_gyri_parts[triangle[2]].append(triangle[0])

        if sulc_data[triangle[1]] < 0 and sulc_data[triangle[2]] < 0:
            if triangle[2] not in point_connect_points_dict_gyri_parts[triangle[1]]:
                point_connect_points_dict_gyri_parts[triangle[1]].append(triangle[2])
            if triangle[1] not in point_connect_points_dict_gyri_parts[triangle[2]]:
                point_connect_points_dict_gyri_parts[triangle[2]].append(triangle[1])
    return point_connect_points_dict_gyri_parts

def create_gyri_connection_count(point_connect_points_dict_thin_gyri_parts):
    gyri_connection_count = dict()
    for point in point_connect_points_dict_thin_gyri_parts.keys():
        count = len(point_connect_points_dict_thin_gyri_parts[point])
        gyri_connection_count[point] = count
    return gyri_connection_count

def find_the_max_patchSize_point(unconnected_points_list, point_patchSize_dict_updated):
    # max_patchSize_point = max(point_patchSize_dict.items(), key=lambda x: x[1])[0]
    max = -1
    for point in unconnected_points_list:
        if point_patchSize_dict_updated[point] > max:
            max = point_patchSize_dict_updated[point]
            max_patchSize_point = point
    return max_patchSize_point

def find_the_max_patchSize_candidate_connection(point_patchSize_dict_updated, connected_candidate_dict):
    max = -1
    candidate_connection_list = list()
    for point in connected_candidate_dict.keys():
        if point_patchSize_dict_updated[point] > max:
            max = point_patchSize_dict_updated[point]
            candidate_connection_list = connected_candidate_dict[point]
        elif point_patchSize_dict_updated[point] == max:
            candidate_connection_list = candidate_connection_list + connected_candidate_dict[point]
    if len(candidate_connection_list) == 1:
        candidate_connection = candidate_connection_list[0]
    else:
        max = -1
        for candidate in candidate_connection_list:
            if point_patchSize_dict_updated[candidate[1]] > max:
                max = point_patchSize_dict_updated[candidate[1]]
                candidate_connection = candidate

    return candidate_connection

def create_tree(orig_sphere_polydata, orig_surf_polydata, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, thin_sulc_data, output_prefix,sphere):
    print('================= create tree:\t' + time.asctime(time.localtime(time.time())) + '=========================')
    point_num = orig_sphere_polydata.GetNumberOfPoints()

    connected_points_list = list()
    connected_lines_list = list()
    connected_candidate_dict = defaultdict(list)
    unconnected_points_list = list()
    father_dict = dict()
    gyri_connection_count = create_gyri_connection_count(point_connect_points_dict_thin_gyri_parts)

    for point in range(point_num):
        if thin_sulc_data[point] < 0:
            unconnected_points_list.append(point)

    debug_counter = 0

    while len(unconnected_points_list) != 0:
        print(' remaining points:\t', len(unconnected_points_list))
        if not connected_candidate_dict:
            # pdb.set_trace()
            max_patchSize_point = find_the_max_patchSize_point(unconnected_points_list, point_patchSize_dict_updated)
            connected_points_list.append(max_patchSize_point)
            unconnected_points_list.remove(max_patchSize_point)
            father_dict[max_patchSize_point] = max_patchSize_point
        else:
            connected_candidate_dict = {}

        available_connected_points_list = [point for point in connected_points_list if gyri_connection_count[point] > 0]
        for point in available_connected_points_list:
            # if gyri_connection_count[point]:
            neighbors = [neighbor for neighbor in point_connect_points_dict_thin_gyri_parts[point] if
                         neighbor not in connected_points_list]
            for neighbor in neighbors:
                if neighbor not in connected_candidate_dict.keys():
                    connected_candidate_dict[neighbor] = [[neighbor, point]]
                else:
                    connected_candidate_dict[neighbor].append([neighbor, point])

        if connected_candidate_dict:
            candidate = find_the_max_patchSize_candidate_connection(point_patchSize_dict_updated, connected_candidate_dict)
            connected_points_list.append(candidate[0])
            unconnected_points_list.remove(candidate[0])
            connected_lines_list.append(candidate)
            # pdb.set_trace()
            for connection_of_candidate in point_connect_points_dict_thin_gyri_parts[candidate[0]]:
                gyri_connection_count[connection_of_candidate] = gyri_connection_count[connection_of_candidate] - 1
            if candidate[0] not in father_dict.keys():
                father_dict[candidate[0]] = candidate[1]
            else:
                print('reconnecting the existing point!')
                exit()
            debug_counter += 1
            # if debug_counter % 1000 == 0:
            #     write_skelenton_by_connectionPair(orig_sphere_polydata, connected_lines_list, output_prefix + '_sphere_skeleton_round(' + str(debug_counter) + ').vtk')
    for connection in connected_lines_list:
        if [connection[1], connection[0]] in connected_lines_list:
            print('connection already in connected_lines_list')
            exit()

    sphere_output = os.path.join(output_prefix, "%s_sphere_inital_tree.vtk"%(sphere))
    surf_output = os.path.join(output_prefix, "%s_surf_inital_tree.vtk"%(sphere))
    write_skelenton_by_connectionPair(orig_sphere_polydata, connected_lines_list, sphere_output)
    write_skelenton_by_connectionPair(orig_surf_polydata, connected_lines_list, surf_output)

    return connected_lines_list, father_dict

if __name__=="__main__":
    print("True")