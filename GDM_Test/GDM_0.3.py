import vtk
import nibabel.freesurfer.io as io
from collections import defaultdict
import pdb
import numpy as np
import os
import shutil
import time
import networkx as nx
import argparse
from utils import featured_sphere


def read_vtk_file(vtk_file):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    Header = reader.GetHeader()

    polydata = reader.GetOutput()

    nCells = polydata.GetNumberOfCells()
    nPolys = polydata.GetNumberOfPolys()
    nLines = polydata.GetNumberOfLines()
    nStrips = polydata.GetNumberOfStrips()
    nPieces = polydata.GetNumberOfPieces()
    nVerts = polydata.GetNumberOfVerts()
    nPoints = polydata.GetNumberOfPoints()
    Points = polydata.GetPoints()
    Point = polydata.GetPoints().GetPoint(0)

    return polydata


def get_connect_points(surf_polydata):
    point_connect_points_dict = defaultdict(list)
    cell_num = surf_polydata.GetNumberOfCells()
    CellArray = surf_polydata.GetPolys()
    Polygons = CellArray.GetData()
    for i in range(0, cell_num):
        triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
        if triangle[1] not in point_connect_points_dict[triangle[0]]:
            point_connect_points_dict[triangle[0]].append(triangle[1])
        if triangle[2] not in point_connect_points_dict[triangle[0]]:
            point_connect_points_dict[triangle[0]].append(triangle[2])
        if triangle[0] not in point_connect_points_dict[triangle[1]]:
            point_connect_points_dict[triangle[1]].append(triangle[0])
        if triangle[2] not in point_connect_points_dict[triangle[1]]:
            point_connect_points_dict[triangle[1]].append(triangle[2])
        if triangle[0] not in point_connect_points_dict[triangle[2]]:
            point_connect_points_dict[triangle[2]].append(triangle[0])
        if triangle[1] not in point_connect_points_dict[triangle[2]]:
            point_connect_points_dict[triangle[2]].append(triangle[1])
    return point_connect_points_dict


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


def delete_isolated_point(point_num, point_neighbor_points_dict, sulc_data):
    marginal_points, marginal_points_gyri, marginal_points_sulc = find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, sulc_data)
    for point in range(point_num):
        sulc = sulc_data[point]
        '''isolated point'''
        nerighbor_sulcs = sulc_data[point_neighbor_points_dict[point]]
        if sulc > 0 and np.sum(nerighbor_sulcs < 0) == nerighbor_sulcs.shape[0]:
            sulc_data[point] = -1
        elif sulc < 0 and np.sum(nerighbor_sulcs > 0) == nerighbor_sulcs.shape[0]:
            sulc_data[point] = 1
        else:
            '''isolated one-round patch'''
            if sulc > 0:
                first_neighbor_list = [first_neighbor for first_neighbor in point_neighbor_points_dict[point] if first_neighbor not in marginal_points_gyri]
            elif sulc < 0:
                first_neighbor_list = [first_neighbor for first_neighbor in point_neighbor_points_dict[point] if first_neighbor not in marginal_points_sulc]

            second_neighbor_list = list()
            for first_neighbor in first_neighbor_list:
                second_neighbors = [second_neighbor for second_neighbor in point_neighbor_points_dict[first_neighbor] if second_neighbor not in [point] + first_neighbor_list]
                second_neighbor_list = second_neighbor_list + second_neighbors
            second_neighbor_list = list(set(second_neighbor_list))

            if sulc > 0 and np.sum(sulc_data[second_neighbor_list] < 0) == len(second_neighbor_list):
                sulc_data[[point] + first_neighbor_list] = -1
            elif sulc < 0 and np.sum(sulc_data[second_neighbor_list] > 0) == len(second_neighbor_list):
                sulc_data[[point] + first_neighbor_list] = 1

    return sulc_data


def delete_isolated_point1(point_num, point_neighbor_points_dict, sulc_data):
    for point in range(point_num):
        sulc = sulc_data[point]
        '''isolated point'''
        nerighbor_sulcs = sulc_data[point_neighbor_points_dict[point]]
        if sulc > 0 and np.sum(nerighbor_sulcs < 0) == nerighbor_sulcs.shape[0]:
            sulc_data[point] = -1
        elif sulc < 0 and np.sum(nerighbor_sulcs > 0) == nerighbor_sulcs.shape[0]:
            sulc_data[point] = 1
        '''isolated one-round patch'''
        first_neighbor_list = point_neighbor_points_dict[point]
        second_neighbor_list = list()
        for first_neighbor in first_neighbor_list:
            neighbors_of_first_round = point_neighbor_points_dict[first_neighbor]
            for neighbor_of_first_round in neighbors_of_first_round:
                if neighbor_of_first_round not in [point] + first_neighbor_list:
                    second_neighbor_list.append(neighbor_of_first_round)
        second_neighbor_list = list(set(second_neighbor_list))
        if sulc > 0 and np.sum(sulc_data[second_neighbor_list] < 0) == len(second_neighbor_list):
            sulc_data[[point] + first_neighbor_list] = -1
        elif sulc < 0 and np.sum(sulc_data[second_neighbor_list] > 0) == len(second_neighbor_list):
            sulc_data[[point] + first_neighbor_list] = 1

    return sulc_data


def find_marginal_point(points_list, point_neighbor_points_dict, sulc_data):
    marginal_points = list()
    marginal_points_gyri = list()
    marginal_points_sulc = list()
    for point in points_list:
        neighbor_points = point_neighbor_points_dict[point]
        neighbor_sulcs = sulc_data[neighbor_points]
        if np.sum(neighbor_sulcs < 0) != len(neighbor_points) and np.sum(neighbor_sulcs > 0) != len(neighbor_points):
            marginal_points.append(point)
            if sulc_data[point] < 0:
                marginal_points_gyri.append(point)
            else:
                marginal_points_sulc.append(point)

    return marginal_points, marginal_points_gyri, marginal_points_sulc


def featured_sphere(orig_sphere_polydata, feature_file_dict, output):
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    cell_num = orig_sphere_polydata.GetNumberOfCells()

    f = open(output, 'w')
    f.write("# vtk DataFile Version 3.0\n")
    f.write("mesh surface\n")
    f.write("ASCII\n")
    f.write("DATASET POLYDATA\n")
    f.write("POINTS " + str(point_num) + " float\n")

    for point in range(point_num):
        coordinate = orig_sphere_polydata.GetPoints().GetPoint(point)
        f.write(str(coordinate[0]) + " " + str(coordinate[1]) + " " + str(coordinate[2]) + '\n')

    f.write("POLYGONS " + str(cell_num) + " " + str(4 * cell_num) + '\n')
    CellArray = orig_sphere_polydata.GetPolys()
    Polygons = CellArray.GetData()
    for i in range(0, cell_num):
        triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
        f.write(str(3) + " " + str(triangle[0]) + " " + str(triangle[1]) + " " + str(triangle[2]) + '\n')

    f.write("POINT_DATA " + str(point_num) + '\n')
    scale_dict = {'sulc': 0.0, 'curv': 0.0, 'thickness': 0.00000000001}
    for feature_name in scale_dict.keys():
        feature_file = feature_file_dict[feature_name]
        features = io.read_morph_data(feature_file)

        f.write("SCALARS " + feature_name + '_binary' + " float" + '\n')
        f.write("LOOKUP_TABLE " + feature_name + '_binary' + '\n')

        for point in range(point_num):
            if features[point] >= scale_dict[feature_name]:
                f.write(str(1) + '\n')
            elif features[point] < -1*scale_dict[feature_name]:
                f.write(str(-1) + '\n')
            else:
                f.write(str(0) + '\n')

        f.write("SCALARS " + feature_name + " float" + '\n')
        f.write("LOOKUP_TABLE " + feature_name + '\n')

        for point in range(point_num):
            f.write(str(features[point]) + '\n')

    f.close()


def write_featured_sphere_from_variable(orig_sphere_polydata, feature_name_variable_dict, output):
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    cell_num = orig_sphere_polydata.GetNumberOfCells()

    f = open(output, 'w')
    f.write("# vtk DataFile Version 3.0\n")
    f.write("mesh surface\n")
    f.write("ASCII\n")
    f.write("DATASET POLYDATA\n")
    f.write("POINTS " + str(point_num) + " float\n")

    for point in range(point_num):
        coordinate = orig_sphere_polydata.GetPoints().GetPoint(point)
        f.write(str(coordinate[0]) + " " + str(coordinate[1]) + " " + str(coordinate[2]) + '\n')

    f.write("POLYGONS " + str(cell_num) + " " + str(4 * cell_num) + '\n')
    CellArray = orig_sphere_polydata.GetPolys()
    Polygons = CellArray.GetData()
    for i in range(0, cell_num):
        triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
        f.write(str(3) + " " + str(triangle[0]) + " " + str(triangle[1]) + " " + str(triangle[2]) + '\n')

    f.write("POINT_DATA " + str(point_num) + '\n')
    for feature_name in feature_name_variable_dict.keys():
        feature_variable = feature_name_variable_dict[feature_name]
        f.write("SCALARS " + feature_name + " float" + '\n')
        f.write("LOOKUP_TABLE " + feature_name + '\n')

        for point in range(point_num):
            f.write(str(feature_variable[point]) + '\n')
    f.close()


def write_featured_sphere_from_variable_single(orig_sphere_polydata, feature_name, feature_variable, output):
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    cell_num = orig_sphere_polydata.GetNumberOfCells()

    f = open(output, 'w')
    f.write("# vtk DataFile Version 3.0\n")
    f.write("mesh surface\n")
    f.write("ASCII\n")
    f.write("DATASET POLYDATA\n")
    f.write("POINTS " + str(point_num) + " float\n")

    for point in range(point_num):
        coordinate = orig_sphere_polydata.GetPoints().GetPoint(point)
        f.write(str(coordinate[0]) + " " + str(coordinate[1]) + " " + str(coordinate[2]) + '\n')

    f.write("POLYGONS " + str(cell_num) + " " + str(4 * cell_num) + '\n')
    CellArray = orig_sphere_polydata.GetPolys()
    Polygons = CellArray.GetData()
    for i in range(0, cell_num):
        triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
        f.write(str(3) + " " + str(triangle[0]) + " " + str(triangle[1]) + " " + str(triangle[2]) + '\n')

    f.write("POINT_DATA " + str(point_num) + '\n')
    f.write("SCALARS " + feature_name + " float" + '\n')
    f.write("LOOKUP_TABLE " + feature_name + '\n')

    for point in range(point_num):
        f.write(str(feature_variable[point]) + '\n')
    f.close()


def find_round_n_neighbor(point, n, point_neighbor_points_dict):
    neighbor_list = list()
    current_outer_points_list = list()
    current_outer_points_list.append(point)
    while n > 0:
        neighbor_list = list(set(neighbor_list + current_outer_points_list))
        next_outer_points_list = list()
        for point in current_outer_points_list:
            neighbors = [neighbor for neighbor in point_neighbor_points_dict[point] if neighbor not in neighbor_list]
            next_outer_points_list = next_outer_points_list + neighbors
        next_outer_points_list = list(set(next_outer_points_list))
        current_outer_points_list = next_outer_points_list
        n = n - 1
    neighbor_list = list(set(neighbor_list + current_outer_points_list))
    return neighbor_list, current_outer_points_list


def find_missing_gyri_by_sulc_and_curv(sulc_data, curv_data, outer_gyri_curv_thres):
    missing_gyri_point_list = list()
    for point in range(sulc_data.shape[0]):
        if sulc_data[point] >= 0 and curv_data[point] < outer_gyri_curv_thres:
            missing_gyri_point_list.append(point)
    return missing_gyri_point_list


def find_inner_sulci_by_sulc_and_curv(sulc_data, curv_data, point_neighbor_points_dict, inner_sulci_curv_thres, inner_sulci_round_thres, inner_sulci_neighbor_curv_thres):
    inner_sulci_point_list = list()
    for point in range(sulc_data.shape[0]):
        if sulc_data[point] < 0 and curv_data[point] > inner_sulci_curv_thres:
            neighbor_collection, round_n_neighbors = find_round_n_neighbor(point, inner_sulci_round_thres, point_neighbor_points_dict)
            if np.sum(curv_data[round_n_neighbors] > inner_sulci_neighbor_curv_thres) > 0:
                for neighbor in neighbor_collection:
                    if curv_data[neighbor] > 0:
                        inner_sulci_point_list.append(neighbor)
    return inner_sulci_point_list


def update_sulc_data(original_sulc_data, missing_gyri_point_list, inner_sulci_point_list):
    updated_sulc_data = np.zeros(shape=original_sulc_data.shape)
    for point in range(original_sulc_data.shape[0]):
        if point in missing_gyri_point_list:
            updated_sulc_data[point] = -1
        elif point in inner_sulci_point_list:
            updated_sulc_data[point] = 1
        else:
            updated_sulc_data[point] = original_sulc_data[point]
    return updated_sulc_data


def thickness_denoise(point_num, point_neighbor_points_dict, thickness_data_raw):
    thickness_data = thickness_data_raw
    for point in range(point_num):
        if thickness_data[point] == 0:
            step_n = 1
            noise = 0
            while not noise and step_n <= 5:
                neighbor_list, outer_points_list = find_round_n_neighbor(point, step_n, point_neighbor_points_dict)
                if np.sum(thickness_data[outer_points_list] > 0) == len(outer_points_list):
                    thickness_data[neighbor_list] = 1
                    noise = 1
                else:
                    step_n += 1
    return thickness_data


def initialize_sulc_data(orig_sphere_polydata, orig_surf_polydata, feature_file_dict, point_neighbor_points_dict, inner_sulci_curv_thres, inner_sulci_round_thres, outer_gyri_curv_thres, inner_sulci_neighbor_curv_thres, output_prefix):
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    cell_num = orig_sphere_polydata.GetNumberOfCells()
    CellArray = orig_sphere_polydata.GetPolys()
    Polygons = CellArray.GetData()

    print('update the sulc value by curv:\t' + time.asctime(time.localtime(time.time())))

    thickness_data_raw = io.read_morph_data(feature_file_dict['thickness'])
    thickness_data = thickness_denoise(point_num, point_neighbor_points_dict, thickness_data_raw)
    sulc_data_raw = io.read_morph_data(feature_file_dict['sulc'])
    sulc_data_delete_thicknessZero = np.where(thickness_data > 0, sulc_data_raw, 0)
    sulc_data_binary = np.where(sulc_data_delete_thicknessZero >= 0, 1, -1)
    sulc_data = delete_isolated_point(point_num, point_neighbor_points_dict, sulc_data_binary)
    curv_data_raw = io.read_morph_data(feature_file_dict['curv'])
    curv_data_delete_thicknessZero = np.where(thickness_data > 0, curv_data_raw, 0)
    missing_gyri_point_list = find_missing_gyri_by_sulc_and_curv(sulc_data, curv_data_delete_thicknessZero, outer_gyri_curv_thres)
    inner_sulci_point_list = find_inner_sulci_by_sulc_and_curv(sulc_data, curv_data_delete_thicknessZero, point_neighbor_points_dict, inner_sulci_curv_thres, inner_sulci_round_thres, inner_sulci_neighbor_curv_thres)
    updated_sulc_data = update_sulc_data(sulc_data, missing_gyri_point_list, inner_sulci_point_list)
    updated_sulc_data = delete_isolated_point(point_num, point_neighbor_points_dict, updated_sulc_data)

    print('draw updated sulc colorful sphere:\t' + time.asctime(time.localtime(time.time())))
    feature_name_variable_dict = {'thickness_data_raw': thickness_data_raw, 'thickness_de_noise': thickness_data,
                                  'sulc_data_raw': sulc_data_raw,
                                  'sulc_data_delete_thicknessZero': sulc_data_delete_thicknessZero,
                                  'sulc_data_binary': sulc_data_binary, 'sulc_data_de_isolation': sulc_data,
                                  'curv_data_raw': curv_data_raw,
                                  'curv_data_delete_thicknessZero': curv_data_delete_thicknessZero,
                                  'updated_sulc_data': updated_sulc_data}
    write_featured_sphere_from_variable(orig_sphere_polydata, feature_name_variable_dict,
                                        output_prefix + '_sphere_feature_updated.vtk')
    write_featured_sphere_from_variable(orig_surf_polydata, feature_name_variable_dict,
                                        output_prefix + '_surf_feature_updated.vtk')
    return updated_sulc_data, sulc_data, curv_data_delete_thicknessZero

def initialize_sulc_data2(orig_sphere_polydata, orig_surf_polydata, feature_file_dict, point_neighbor_points_dict, inner_sulci_curv_thres, inner_sulci_round_thres, outer_gyri_curv_thres, inner_sulci_neighbor_curv_thres, output_prefix,sulc_threshold):
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    cell_num = orig_sphere_polydata.GetNumberOfCells()
    CellArray = orig_sphere_polydata.GetPolys()
    Polygons = CellArray.GetData()

    print('update the sulc value by curv:\t' + time.asctime(time.localtime(time.time())))

    thickness_data_raw = io.read_morph_data(feature_file_dict['thickness'])
    thickness_data = thickness_denoise(point_num, point_neighbor_points_dict, thickness_data_raw)
    sulc_data_raw = io.read_morph_data(feature_file_dict['sulc'])
    sulc_data_delete_thicknessZero = np.where(thickness_data > 0, sulc_data_raw, 0)
    # sulc_data_binary = np.where(sulc_data_delete_thicknessZero >= 0, 1, -1)
    sulc_data_binary = np.where(sulc_data_delete_thicknessZero >= sulc_threshold, 1, -1)
    sulc_data = delete_isolated_point(point_num, point_neighbor_points_dict, sulc_data_binary)
    curv_data_raw = io.read_morph_data(feature_file_dict['curv'])
    curv_data_delete_thicknessZero = np.where(thickness_data > 0, curv_data_raw, 0)
    missing_gyri_point_list = find_missing_gyri_by_sulc_and_curv(sulc_data, curv_data_delete_thicknessZero, outer_gyri_curv_thres)
    inner_sulci_point_list = find_inner_sulci_by_sulc_and_curv(sulc_data, curv_data_delete_thicknessZero, point_neighbor_points_dict, inner_sulci_curv_thres, inner_sulci_round_thres, inner_sulci_neighbor_curv_thres)
    updated_sulc_data = update_sulc_data(sulc_data, missing_gyri_point_list, inner_sulci_point_list)
    updated_sulc_data = delete_isolated_point(point_num, point_neighbor_points_dict, updated_sulc_data)

    print('draw updated sulc colorful sphere:\t' + time.asctime(time.localtime(time.time())))
    feature_name_variable_dict = {'thickness_data_raw': thickness_data_raw, 'thickness_de_noise': thickness_data,
                                  'sulc_data_raw': sulc_data_raw,
                                  'sulc_data_delete_thicknessZero': sulc_data_delete_thicknessZero,
                                  'sulc_data_binary': sulc_data_binary, 'sulc_data_de_isolation': sulc_data,
                                  'curv_data_raw': curv_data_raw,
                                  'curv_data_delete_thicknessZero': curv_data_delete_thicknessZero,
                                  'updated_sulc_data': updated_sulc_data}
    write_featured_sphere_from_variable(orig_sphere_polydata, feature_name_variable_dict,
                                        output_prefix + '_sphere_feature_updated.vtk')
    write_featured_sphere_from_variable(orig_surf_polydata, feature_name_variable_dict,
                                        output_prefix + '_surf_feature_updated.vtk')
    return updated_sulc_data, sulc_data, curv_data_delete_thicknessZero


def write_thin_gyri_on_sphere_point_marginal_sulc_curv(orig_sphere_polydata, orig_surf_polydata, point_neighbor_points_dict, updated_sulc_data, curv_data_delete_thicknessZero, expend_curv_step_size, output_prefix):
    print('================= begin erosion:\t' + time.asctime(time.localtime(time.time())) + '=========================')
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    cell_num = orig_sphere_polydata.GetNumberOfCells()
    CellArray = orig_sphere_polydata.GetPolys()
    Polygons = CellArray.GetData()

    print('create triangles connection:\t' + time.asctime(time.localtime(time.time())))
    triangle_collection = list()
    for cell in range(cell_num):
        triangle = set([Polygons.GetValue(j) for j in range(cell * 4 + 1, cell * 4 + 4)])
        triangle_collection.append(triangle)

    print('find initial marginal points:\t' + time.asctime(time.localtime(time.time())))
    marginal_points, marginal_points_gyri, marginal_points_sulc = find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, updated_sulc_data)
    # pdb.set_trace()

    min_curv = curv_data_delete_thicknessZero.min()

    print('begin expending:\t' + time.asctime(time.localtime(time.time())))
    round = 0
    current_curv = 0
    curv_step = expend_curv_step_size
    while current_curv >= min_curv:
        current_curv = current_curv - curv_step
        finish = 1
        print('\n================ current curv:', current_curv, '====================')
        while finish > 0:
            round += 1
            print('round', str(round), ':\t', time.asctime(time.localtime(time.time())))
            finish = 0
            redlize_points = list()
            marginal_points_candidate = list()
            for point in marginal_points_sulc:
                sulc = updated_sulc_data[point]
                if sulc < 0:
                    print('updated_sulc_data sulc value error!')
                    exit()
                else:
                    # neighbor_points = point_neighbor_points_dict[point]
                    neighbor_points = [neighbor for neighbor in point_neighbor_points_dict[point]]
                    for neighbor_point in neighbor_points:
                        if updated_sulc_data[neighbor_point] < 0 and curv_data_delete_thicknessZero[neighbor_point] >= current_curv:
                            second_neighbors = point_neighbor_points_dict[neighbor_point]
                            marginal_gyri_num = 0
                            marginal_gyri_points = list()
                            for second_neighbor in second_neighbors:
                                if updated_sulc_data[second_neighbor] < 0 and second_neighbor in marginal_points_gyri:
                                    marginal_gyri_num += 1
                                    marginal_gyri_points.append(second_neighbor)
                            if (marginal_gyri_num < 2) or (marginal_gyri_num == 2 and (np.sum(updated_sulc_data[second_neighbors] < 0) > 2 or set(marginal_gyri_points + [neighbor_point]) in triangle_collection)):
                                redlize_points.append(neighbor_point)
                                marginal_points_candidate = list(set(marginal_points_candidate + second_neighbors))
                                finish = 1
            updated_sulc_data[redlize_points] = 1
            marginal_points, marginal_points_gyri, marginal_points_sulc = find_marginal_point(list(set(marginal_points + marginal_points_candidate)), point_neighbor_points_dict, updated_sulc_data)
            # pdb.set_trace()
            if round % 500 == 0:
                write_featured_sphere_from_variable(orig_sphere_polydata, {'sulc_updated': updated_sulc_data}, output_prefix + '_sphere_round(' + str(round) + ')_marginal_point.vtk')

    print('draw final thin path on sphere:\t' + time.asctime(time.localtime(time.time())))
    write_featured_sphere_from_variable(orig_sphere_polydata, {'sulc_updated': updated_sulc_data}, output_prefix + '_sphere_thin.vtk')
    write_featured_sphere_from_variable(orig_surf_polydata, {'sulc_updated': updated_sulc_data}, output_prefix + '_surf_thin.vtk')
    return updated_sulc_data


def find_the_patchSize_of_gyri_point(sulc_data, point_neighbor_points_dict):
    point_patchSize_dict = dict()
    for point in point_neighbor_points_dict.keys():
        if sulc_data[point] > 0:
            point_patchSize_dict[point] = -1
        else:
            inner_points_list = list()
            outer_points_list = list()
            outer_points_list.append(point)
            round_num = 0
            while np.sum(sulc_data[outer_points_list] < 0) == len(outer_points_list):
                round_num += 1
                inner_points_list = inner_points_list + outer_points_list
                next_outer_candidate = list()
                for outer_point in outer_points_list:
                    neighbor_points = [neighbor_point for neighbor_point in point_neighbor_points_dict[outer_point] if neighbor_point not in inner_points_list]
                    next_outer_candidate = next_outer_candidate + neighbor_points

                outer_points_list = list(set(next_outer_candidate))
            point_patchSize_dict[point] = round_num
    return point_patchSize_dict


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


def write_skelenton_by_connectionPair(orig_sphere_polydata, connected_lines_list, output):
    points_new = vtk.vtkPoints()
    lines_cell_new = vtk.vtkCellArray()
    line_id = 0
    for line in connected_lines_list:
        coordinate1 = orig_sphere_polydata.GetPoints().GetPoint(line[0])
        points_new.InsertNextPoint(coordinate1)
        coordinate2 = orig_sphere_polydata.GetPoints().GetPoint(line[1])
        points_new.InsertNextPoint(coordinate2)

        line = vtk.vtkLine()
        line.GetPointIds().SetNumberOfIds(2)
        line.GetPointIds().SetId(0, line_id*2)
        line.GetPointIds().SetId(1, line_id*2 + 1)
        lines_cell_new.InsertNextCell(line)
        line_id += 1

    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points_new)
    polygonPolyData.SetLines(lines_cell_new)
    polygonPolyData.Modified()

    if vtk.VTK_MAJOR_VERSION <= 5:
        polygonPolyData = polygonPolyData.GetProducerPort()
    else:
        polygonPolyData = polygonPolyData

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polygonPolyData)
    writer.SetFileName(output)
    writer.Write()


def write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, connected_lines_list, output):
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    cell_num = orig_sphere_polydata.GetNumberOfCells()
    points_new = vtk.vtkPoints()
    lines_cell_new = vtk.vtkCellArray()
    for point in range(point_num):
        coordinate = orig_sphere_polydata.GetPoints().GetPoint(point)
        points_new.InsertNextPoint(coordinate)

    for connection in connected_lines_list:
        line = vtk.vtkLine()
        line.GetPointIds().SetNumberOfIds(2)
        line.GetPointIds().SetId(0, connection[0])
        line.GetPointIds().SetId(1, connection[1])
        lines_cell_new.InsertNextCell(line)

    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points_new)
    polygonPolyData.SetLines(lines_cell_new)
    polygonPolyData.Modified()

    if vtk.VTK_MAJOR_VERSION <= 5:
        polygonPolyData = polygonPolyData.GetProducerPort()
    else:
        polygonPolyData = polygonPolyData

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polygonPolyData)
    writer.SetFileName(output)
    writer.Write()


def create_gyri_connection_count(point_connect_points_dict_thin_gyri_parts):
    gyri_connection_count = dict()
    for point in point_connect_points_dict_thin_gyri_parts.keys():
        count = len(point_connect_points_dict_thin_gyri_parts[point])
        gyri_connection_count[point] = count
    return gyri_connection_count


def create_tree(orig_sphere_polydata, orig_surf_polydata, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, thin_sulc_data, output_prefix):
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

    write_skelenton_by_connectionPair(orig_sphere_polydata, connected_lines_list, output_prefix + '_sphere_initial_tree.vtk')
    write_skelenton_by_connectionPair(orig_surf_polydata, connected_lines_list, output_prefix + '_surf_initial_tree.vtk')

    return connected_lines_list, father_dict


def find_skelenton(orig_sphere_polydata, orig_surf_polydata, point_patchSize_dict_updated, curv_data_delete_thicknessZero, original_sulc_data, thin_sulc_data, point_neighbor_points_dict, point_connect_points_dict_thin_gyri_parts, connected_lines_list, father_dict, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix):
    print('================= build skeleton:\t' + time.asctime(time.localtime(time.time())) + '=======================')
    marginal_points, marginal_points_gyri, marginal_points_sulc = find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, thin_sulc_data)

    print('begin trimming  \t' + time.asctime(time.localtime(time.time())))
    tree_point_lines_dict = create_tree_connection_dict(connected_lines_list)
    inner_gyri_points = [gyri_point for gyri_point in tree_point_lines_dict if gyri_point not in marginal_points_gyri]
    initial_skelenton_point_lines_dict, deleted_round1_points = trim_allRound1_long(tree_point_lines_dict, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, father_dict, inner_gyri_points)
    initial_skelenton_connection_list = create_connectingPair(initial_skelenton_point_lines_dict)
    print('draw initial skelenton  \t' + time.asctime(time.localtime(time.time())))
    write_skelenton_by_connectionPair(orig_sphere_polydata, initial_skelenton_connection_list, output_prefix + '_sphere_initial_skelenton.vtk')
    write_skelenton_by_connectionPair(orig_surf_polydata, initial_skelenton_connection_list, output_prefix + '_surf_initial_skelenton.vtk')

    print('connecting breaks in skelenton  \t' + time.asctime(time.localtime(time.time())))
    final_point_lines_dict = connect_break_in_circle(initial_skelenton_point_lines_dict, point_connect_points_dict_thin_gyri_parts, deleted_round1_points, father_dict)
    final_connection_list = create_connectingPair(final_point_lines_dict)
    write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, final_connection_list, output_prefix + '_sphere_skelenton_connect_break_allpoints.vtk')
    write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata, final_connection_list, output_prefix + '_surf_skelenton_connect_break_allpoints.vtk')

    final_point_lines_dict = clear_empty_point_in_dict(final_point_lines_dict)

    print('find missing gyri  \t' + time.asctime(time.localtime(time.time())))
    final_point_lines_dict = create_connection_for_missing_gyri(orig_surf_polydata, orig_sphere_polydata, final_point_lines_dict, original_sulc_data, curv_data_delete_thicknessZero, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix)
    all_connection_list = create_connectingPair(final_point_lines_dict)
    write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, all_connection_list, output_prefix + '_sphere_skelenton_allpoints_final.vtk')
    write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata, all_connection_list, output_prefix + '_surf_skelenton_allpoints_final.vtk')

    all_point_lines_dict = create_tree_connection_dict(all_connection_list)

    print('find and draw 3hinge' + time.asctime(time.localtime(time.time())))
    hinge3_list = create_3hinge(all_point_lines_dict)
    hinge3_txt = open(output_prefix + '_3hinge_ids.txt', "w")
    for hinge in hinge3_list:
        hinge3_txt.write(str(hinge) + "\n")
    hinge3_txt.close()
    draw_3hinge_on_surf(orig_sphere_polydata, hinge3_list, output_prefix + '_sphere_3hinge_vertex.vtk')
    draw_3hinge_on_surf(orig_surf_polydata, hinge3_list, output_prefix + '_surf_3hinge_vertex.vtk')


def find_skelenton_missing(orig_sphere_polydata, orig_surf_polydata, skeleton_polydata, curv_data_delete_thicknessZero, original_sulc_data, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix):
    print('================= build skeleton:\t' + time.asctime(time.localtime(time.time())) + '=======================')
    final_connection_list = read_connection_of_skelenton_file(skeleton_polydata)
    final_point_lines_dict = create_tree_connection_dict(final_connection_list)
    print('find missing gyri  \t' + time.asctime(time.localtime(time.time())))
    final_point_lines_dict = create_connection_for_missing_gyri(orig_surf_polydata, orig_sphere_polydata,
                                                                final_point_lines_dict, original_sulc_data,
                                                                curv_data_delete_thicknessZero,
                                                                length_thres_of_long_gyri,
                                                                neighbor_missing_path_smallest_step,
                                                                flat_threshold_for_convex_gyri, nearest_skeleton_num,
                                                                island_gyri_length_thres, output_prefix)
    all_connection_list = create_connectingPair(final_point_lines_dict)
    write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, all_connection_list, output_prefix + '_sphere_skelenton_allpoints_final.vtk')
    write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata, all_connection_list, output_prefix + '_surf_skelenton_allpoints_final.vtk')

    all_point_lines_dict = create_tree_connection_dict(all_connection_list)

    print('find and draw 3hinge' + time.asctime(time.localtime(time.time())))
    hinge3_list = create_3hinge(all_point_lines_dict)
    hinge3_txt = open(output_prefix + '_3hinge_ids.txt', "w")
    for hinge in hinge3_list:
        hinge3_txt.write(str(hinge) + "\n")
    hinge3_txt.close()
    draw_3hinge_on_surf(orig_sphere_polydata, hinge3_list, output_prefix + '_sphere_3hinge_vertex.vtk')
    draw_3hinge_on_surf(orig_surf_polydata, hinge3_list, output_prefix + '_surf_3hinge_vertex.vtk')

def find_skelenton_missing2(orig_sphere_polydata, orig_surf_polydata, skeleton_polydata, curv_data_delete_thicknessZero, original_sulc_data, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix):
    print('================= build skeleton:\t' + time.asctime(time.localtime(time.time())) + '=======================')
    final_connection_list = read_connection_of_skelenton_file(skeleton_polydata)
    final_point_lines_dict = create_tree_connection_dict(final_connection_list)
    print('find missing gyri  \t' + time.asctime(time.localtime(time.time())))
    final_point_lines_dict = create_connection_for_missing_gyri(orig_surf_polydata, orig_sphere_polydata,
                                                                final_point_lines_dict, original_sulc_data,
                                                                curv_data_delete_thicknessZero,
                                                                length_thres_of_long_gyri,
                                                                neighbor_missing_path_smallest_step,
                                                                flat_threshold_for_convex_gyri, nearest_skeleton_num,
                                                                island_gyri_length_thres, output_prefix)
    all_connection_list = create_connectingPair(final_point_lines_dict)
    write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, all_connection_list, output_prefix + '_sphere_skelenton_allpoints_final_GDM.vtk')
    write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata, all_connection_list, output_prefix + '_surf_skelenton_allpoints_final_GDM.vtk')

    all_point_lines_dict = create_tree_connection_dict(all_connection_list)

    print('find and draw 3hinge' + time.asctime(time.localtime(time.time())))
    hinge3_list = create_3hinge(all_point_lines_dict)
    hinge3_txt = open(output_prefix + '_3hinge_ids.txt', "w")
    for hinge in hinge3_list:
        hinge3_txt.write(str(hinge) + "\n")
    hinge3_txt.close()
    draw_3hinge_on_surf(orig_sphere_polydata, hinge3_list, output_prefix + '_sphere_3hinge_vertex_GDM.vtk')
    draw_3hinge_on_surf(orig_surf_polydata, hinge3_list, output_prefix + '_surf_3hinge_vertex_GDM.vtk')


def read_connection_of_skelenton_file(skelenten_polydata):
    connected_lines_list = list()
    line_num = skelenten_polydata.GetNumberOfCells()
    for line in range(line_num):
        pointID1 = skelenten_polydata.GetCell(line).GetPointIds().GetId(0)
        pointID2 = skelenten_polydata.GetCell(line).GetPointIds().GetId(1)
        connected_lines_list.append([pointID1, pointID2])
    return connected_lines_list


def find_convex_marginal_points_of_long_gyri(orig_sphere_polydata, final_point_lines_dict, point_neighbor_points_dict, point_connect_points_dict_gyri_parts, marginal_points_gyri, original_sulc_data, curv_data_delete_thicknessZero, point_patchSize_dict_orig, length_thres_of_long_gyri, nearest_skeleton_num, island_gyri_length_thres):
    print('find convex marginal points of long gyri  \t' + time.asctime(time.localtime(time.time())))
    convex_marginal_points_gyri = list()
    for point in marginal_points_gyri:
        first_gyri_neighbor = [first_neighbor for first_neighbor in point_neighbor_points_dict[point] if original_sulc_data[first_neighbor] < 0]
        first_sulc_neighbor = [first_neighbor for first_neighbor in point_neighbor_points_dict[point] if original_sulc_data[first_neighbor] > 0]
        if (len(first_gyri_neighbor) - 2 < len(first_sulc_neighbor)) and curv_data_delete_thicknessZero[point] < 0:
            convex_marginal_points_gyri.append(point)
    convex_marginal_points_length_dict = defaultdict(list)
    long_convax_marginal_points_list = list()
    island_convex_marginal_points_length_dict = defaultdict(list)
    for convex_marginal_point in convex_marginal_points_gyri:
        from_marginal_to_skelenton_father_sons_dict = defaultdict(list)
        from_skelenton_to_marginal_son_fathers_dict = defaultdict(list)
        path = 0
        reach_skelenton = 0
        max_step_after_reach_skelenton = 6
        skelenton_point_set = set()
        current_outer_points_list = list()
        current_outer_points_list.append(convex_marginal_point)
        while max_step_after_reach_skelenton and (not reach_skelenton):
            next_outer_points_list = list()
            for point in current_outer_points_list:
                neighbors = [neighbor for neighbor in point_connect_points_dict_gyri_parts[point] if neighbor not in from_marginal_to_skelenton_father_sons_dict.keys() and neighbor not in current_outer_points_list and curv_data_delete_thicknessZero[neighbor] < 0]
                if point in from_marginal_to_skelenton_father_sons_dict.keys():
                    print('from_marginal_to_skelenton_father_sons_dict error!')
                else:
                    from_marginal_to_skelenton_father_sons_dict[point] = neighbors
                for neighbor in neighbors:
                    if neighbor not in from_skelenton_to_marginal_son_fathers_dict.keys():
                        from_skelenton_to_marginal_son_fathers_dict[neighbor] = [point]
                    else:
                        from_skelenton_to_marginal_son_fathers_dict[neighbor].append(point)

                next_outer_points_list = next_outer_points_list + neighbors

            if len(next_outer_points_list):

                path = path + 1

                # test single point
                # if convex_marginal_point == 71363:
                #     draw_3hinge_on_surf(orig_sphere_polydata, next_outer_points_list,
                #                         output_prefix + '_71363_path_' + str(path) + '.vtk')
                #     print(path)

                skelenton_point_set = skelenton_point_set | (set(next_outer_points_list) & set(final_point_lines_dict.keys()))
                if skelenton_point_set:
                    if max_step_after_reach_skelenton == 6:
                        real_path = path
                        averaged_patchSize = 0
                        for skelenton_point in skelenton_point_set:
                            averaged_patchSize = averaged_patchSize + point_patchSize_dict_orig[skelenton_point]
                        averaged_patchSize = averaged_patchSize / len(skelenton_point_set)
                    max_step_after_reach_skelenton -= 1
                if len(skelenton_point_set) >= nearest_skeleton_num:
                    reach_skelenton = 1
                    if real_path - averaged_patchSize > length_thres_of_long_gyri:
                        convex_marginal_points_length_dict[convex_marginal_point] = [real_path, from_marginal_to_skelenton_father_sons_dict, from_skelenton_to_marginal_son_fathers_dict]
                        long_convax_marginal_points_list.append(convex_marginal_point)
                else:
                    current_outer_points_list = set(next_outer_points_list)
            else:
                if path > island_gyri_length_thres:
                    island_convex_marginal_points_length_dict[convex_marginal_point] = [path, from_marginal_to_skelenton_father_sons_dict, from_skelenton_to_marginal_son_fathers_dict]
                break

    return long_convax_marginal_points_list, convex_marginal_points_length_dict, island_convex_marginal_points_length_dict


def select_convex_endpoint_for_each_missing_gyri(long_path_convex_points_list, convex_marginal_points_length_dict, curv_data_delete_thicknessZero, point_connect_points_dict_gyri_parts, neighbor_missing_path_smallest_step):
    print('select convex endpoint for each missing gyri \t' + time.asctime(time.localtime(time.time())))
    delete_list = list()
    for convex_point in long_path_convex_points_list:
        point_connection = [point for point in convex_marginal_points_length_dict[convex_point][1].keys() if point in long_path_convex_points_list]
        current_point = point_connection[0]
        max_path = convex_marginal_points_length_dict[current_point][0]
        step_n_neighbor_list, _ = find_round_n_neighbor(convex_point, neighbor_missing_path_smallest_step, point_connect_points_dict_gyri_parts)
        for point in point_connection:
            if point in step_n_neighbor_list:
                path = convex_marginal_points_length_dict[point][0]
                if path < max_path:
                    delete_list.append(point)
                elif path > max_path:
                    max_path = path
                    delete_list.append(current_point)
                    current_point = point
                else:
                    if curv_data_delete_thicknessZero[point] < curv_data_delete_thicknessZero[current_point]:
                        delete_list.append(current_point)
                        current_point = point
    delete_list = list(set(delete_list))
    for point in delete_list:
        long_path_convex_points_list.remove(point)
    return long_path_convex_points_list


def build_point_id_graphIndex_mapping(convex_point, from_skelenton_to_marginal_son_fathers_dict):
    index = 0
    pointId_graphIndex_dict = dict()
    graphIndex_pointId_dict = dict()
    pointId_graphIndex_dict[convex_point] = index
    graphIndex_pointId_dict[index] = convex_point
    for point in from_skelenton_to_marginal_son_fathers_dict.keys():
        index += 1
        pointId_graphIndex_dict[point] = index
        graphIndex_pointId_dict[index] = point
    return pointId_graphIndex_dict, graphIndex_pointId_dict


def build_graph_for_convex_point(convex_point, id_graphIndex_dict, from_marginal_to_skelenton_father_sons_dict, curv_data_delete_thicknessZero):
    graph = np.full((len(id_graphIndex_dict.keys()), len(id_graphIndex_dict.keys())), 100000000, dtype='float')
    for father_point in from_marginal_to_skelenton_father_sons_dict.keys():
        for son_point in from_marginal_to_skelenton_father_sons_dict[father_point]:
            # weight = 1/(abs(curv_data_delete_thicknessZero[son_point]) + 0.000000001)
            weight = 1 / (pow(curv_data_delete_thicknessZero[son_point], 2) + 0.000000001)
            graph[id_graphIndex_dict[father_point], id_graphIndex_dict[son_point]] = weight
            graph[id_graphIndex_dict[son_point], id_graphIndex_dict[father_point]] = weight
    return graph


def find_connections_for_long_convex_marginal_point(long_convex_marginal_point, final_point_lines_dict, point_connect_points_dict_gyri_parts, curv_data_delete_thicknessZero):
    from_marginal_to_skelenton_father_sons_dict = defaultdict(list)
    from_skelenton_to_marginal_son_fathers_dict = defaultdict(list)
    connection_dicts_list = list()
    reach_skelenton = 0
    skelenton_point_set = set()
    current_outer_points_list = list()
    current_outer_points_list.append(long_convex_marginal_point)
    while not reach_skelenton:
        next_outer_points_list = list()
        for point in current_outer_points_list:
            neighbors = [neighbor for neighbor in point_connect_points_dict_gyri_parts[point] if
                         neighbor not in from_marginal_to_skelenton_father_sons_dict.keys() and neighbor not in current_outer_points_list and
                         curv_data_delete_thicknessZero[neighbor] < 0]
            if point in from_marginal_to_skelenton_father_sons_dict.keys():
                print('from_marginal_to_skelenton_father_sons_dict error!')
            else:
                from_marginal_to_skelenton_father_sons_dict[point] = neighbors
            for neighbor in neighbors:
                if neighbor not in from_skelenton_to_marginal_son_fathers_dict.keys():
                    from_skelenton_to_marginal_son_fathers_dict[neighbor] = [point]
                else:
                    from_skelenton_to_marginal_son_fathers_dict[neighbor].append(point)

            next_outer_points_list = next_outer_points_list + neighbors

        if len(next_outer_points_list):
            skelenton_point_set = skelenton_point_set | (
                        set(next_outer_points_list) & set(final_point_lines_dict.keys()))
            if len(skelenton_point_set) >= 6:
                reach_skelenton = 1
                connection_dicts_list.append(from_marginal_to_skelenton_father_sons_dict)
                connection_dicts_list.append(from_skelenton_to_marginal_son_fathers_dict)

            else:
                current_outer_points_list = set(next_outer_points_list)
        else:
            break
    return connection_dicts_list


def find_connections_for_long_island_convex_marginal_point(long_convex_marginal_point, point_connect_points_dict_gyri_parts, curv_data_delete_thicknessZero):
    from_marginal_to_skelenton_father_sons_dict = defaultdict(list)
    from_skelenton_to_marginal_son_fathers_dict = defaultdict(list)
    connection_dicts_list = list()
    reach_end = 0
    current_outer_points_list = list()
    current_outer_points_list.append(long_convex_marginal_point)
    while not reach_end:
        next_outer_points_list = list()
        for point in current_outer_points_list:
            neighbors = [neighbor for neighbor in point_connect_points_dict_gyri_parts[point] if
                         neighbor not in from_marginal_to_skelenton_father_sons_dict.keys() and neighbor not in current_outer_points_list and
                         curv_data_delete_thicknessZero[neighbor] < 0]
            if point in from_marginal_to_skelenton_father_sons_dict.keys():
                print('from_marginal_to_skelenton_father_sons_dict error!')
            else:
                from_marginal_to_skelenton_father_sons_dict[point] = neighbors
            for neighbor in neighbors:
                if neighbor not in from_skelenton_to_marginal_son_fathers_dict.keys():
                    from_skelenton_to_marginal_son_fathers_dict[neighbor] = [point]
                else:
                    from_skelenton_to_marginal_son_fathers_dict[neighbor].append(point)

            next_outer_points_list = next_outer_points_list + neighbors

        if len(next_outer_points_list):
            current_outer_points_list = set(next_outer_points_list)
        else:
            connection_dicts_list.append(from_marginal_to_skelenton_father_sons_dict)
            connection_dicts_list.append(from_skelenton_to_marginal_son_fathers_dict)
            connection_dicts_list.append(current_outer_points_list)
            reach_end = 1
    return connection_dicts_list


def build_missing_connection_recurrently(long_path_convex_points_list_sorted, final_point_lines_dict, point_connect_points_dict_gyri_parts, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
    missing_connection_list = list()
    for long_convex_marginal_point in long_path_convex_points_list_sorted:
        connection_dicts_list = find_connections_for_long_convex_marginal_point(long_convex_marginal_point,
                                                                                final_point_lines_dict,
                                                                                point_connect_points_dict_gyri_parts,
                                                                                curv_data_delete_thicknessZero)
        if connection_dicts_list:
            from_marginal_to_skelenton_father_sons_dict = connection_dicts_list[0]
            from_skelenton_to_marginal_son_fathers_dict = connection_dicts_list[1]
            pointId_graphIndex_dict, graphIndex_pointId_dict = build_point_id_graphIndex_mapping(long_convex_marginal_point,
                                                                                                 from_skelenton_to_marginal_son_fathers_dict)
            graph = build_graph_for_convex_point(long_convex_marginal_point, pointId_graphIndex_dict,
                                                 from_marginal_to_skelenton_father_sons_dict,
                                                 curv_data_delete_thicknessZero)
            skelenton_point_list = [point for point in from_skelenton_to_marginal_son_fathers_dict.keys() if
                                    point in final_point_lines_dict.keys()]
            shortest_length = 100000000
            shortest_path = []
            for skelenton_point in skelenton_point_list:
                skelenton_graph_id = pointId_graphIndex_dict[skelenton_point]
                long_convex_marginal_point_id = pointId_graphIndex_dict[long_convex_marginal_point]
                path_list, length = create_shortest_path(graph, long_convex_marginal_point_id, skelenton_graph_id)
                if length < shortest_length:
                    shortest_path = path_list
                    shortest_length = length
            if shortest_path:
                if determine_real_long_path(shortest_path, graphIndex_pointId_dict, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
                    for i in range(len(shortest_path) - 1):
                        if [graphIndex_pointId_dict[shortest_path[i]], graphIndex_pointId_dict[shortest_path[i + 1]]] not in missing_connection_list and [graphIndex_pointId_dict[shortest_path[i + 1]], graphIndex_pointId_dict[shortest_path[i]]] not in missing_connection_list:
                            missing_connection_list.append([graphIndex_pointId_dict[shortest_path[i]], graphIndex_pointId_dict[shortest_path[i + 1]]])

                        if graphIndex_pointId_dict[shortest_path[i]] not in final_point_lines_dict.keys():
                            final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i]]] = {graphIndex_pointId_dict[shortest_path[i + 1]]}
                        else:
                            final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i]]].add(graphIndex_pointId_dict[shortest_path[i + 1]])
                        if graphIndex_pointId_dict[shortest_path[i + 1]] not in final_point_lines_dict.keys():
                            final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i + 1]]] = {graphIndex_pointId_dict[shortest_path[i]]}
                        else:
                            final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i + 1]]].add(graphIndex_pointId_dict[shortest_path[i]])
    return missing_connection_list, final_point_lines_dict


def build_main_skeleton_for_missing_island_gyri(longest_island_convex_point, final_point_lines_dict, point_connect_points_dict_gyri_parts, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
    print('build main skeleton for missing island \t' + time.asctime(time.localtime(time.time())))
    current_island_convex_point = longest_island_convex_point
    connection_dicts_list = find_connections_for_long_island_convex_marginal_point(current_island_convex_point,
                                                                                   point_connect_points_dict_gyri_parts,
                                                                                   curv_data_delete_thicknessZero)
    if connection_dicts_list:
        from_marginal_to_skelenton_father_sons_dict = connection_dicts_list[0]
        from_skelenton_to_marginal_son_fathers_dict = connection_dicts_list[1]
        pointId_graphIndex_dict, graphIndex_pointId_dict = build_point_id_graphIndex_mapping(current_island_convex_point,
                                                                                             from_skelenton_to_marginal_son_fathers_dict)
        graph = build_graph_for_convex_point(current_island_convex_point, pointId_graphIndex_dict,
                                             from_marginal_to_skelenton_father_sons_dict,
                                             curv_data_delete_thicknessZero)
        skelenton_point_list = connection_dicts_list[2]
        shortest_length = 100000000
        shortest_path = []
        for skelenton_point in skelenton_point_list:
            skelenton_graph_id = pointId_graphIndex_dict[skelenton_point]
            long_convex_marginal_point_id = pointId_graphIndex_dict[current_island_convex_point]
            path_list, length = create_shortest_path(graph, long_convex_marginal_point_id, skelenton_graph_id)
            if length < shortest_length:
                shortest_path = path_list
                shortest_length = length
        if determine_real_long_path(shortest_path, graphIndex_pointId_dict, point_patchSize_dict_orig,
                                    curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
            for i in range(len(shortest_path) - 1):
                if graphIndex_pointId_dict[shortest_path[i]] not in final_point_lines_dict.keys():
                    final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i]]] = {graphIndex_pointId_dict[shortest_path[i + 1]]}
                else:
                    final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i]]].add(graphIndex_pointId_dict[shortest_path[i + 1]])
                if graphIndex_pointId_dict[shortest_path[i + 1]] not in final_point_lines_dict.keys():
                    final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i + 1]]] = {graphIndex_pointId_dict[shortest_path[i]]}
                else:
                    final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i + 1]]].add(graphIndex_pointId_dict[shortest_path[i]])
    return final_point_lines_dict, connection_dicts_list


def determine_real_long_path_by_curv(path, graphIndex_pointId_dict, curv_data_delete_thicknessZero):
    curv_of_path = list()
    for index in path:
        point = graphIndex_pointId_dict[index]
        curv_of_path.append(curv_data_delete_thicknessZero[point])
    return curv_of_path


def determine_real_long_path(path, graphIndex_pointId_dict, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
    max_patchsize = 0
    for index in path:
        point = graphIndex_pointId_dict[index]
        if point_patchSize_dict_orig[point] > max_patchsize:
            max_patchsize = point_patchSize_dict_orig[point]
    if len(path) < 2 * max_patchsize:
        curv_of_path = determine_real_long_path_by_curv(path, graphIndex_pointId_dict, curv_data_delete_thicknessZero)
        curv_of_path_array = np.stack(curv_of_path)
        if np.sum(curv_of_path_array[0:6] > - 0.1) >= flat_threshold_for_convex_gyri:
            return False
        else:
            return True
    else:
        return True


def sort_the_long_path_convex_point(long_path_convex_points_list, convex_marginal_points_length_dict):
    long_path_convex_points_pathLength_dict = dict()
    for long_convex_point in long_path_convex_points_list:
        long_path_convex_points_pathLength_dict[long_convex_point] = convex_marginal_points_length_dict[long_convex_point][0]
    long_path_convex_points_pathLength_dict_sorted = sorted(long_path_convex_points_pathLength_dict.items(), key=lambda x: x[1], reverse=True)
    long_path_convex_points_list_sorted = [item[0] for item in long_path_convex_points_pathLength_dict_sorted]
    return long_path_convex_points_list_sorted


def create_connection_for_missing_gyri(orig_surf_polydata, orig_sphere_polydata, final_point_lines_dict, original_sulc_data, curv_data_delete_thicknessZero, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix):
    point_connect_points_dict_gyri_parts = get_connect_points_gyri_part(orig_sphere_polydata, original_sulc_data)
    point_neighbor_points_dict = get_connect_points(orig_sphere_polydata)
    point_patchSize_dict_orig = find_the_patchSize_of_gyri_point(original_sulc_data, point_neighbor_points_dict)
    draw_patchSize_colorful_file(orig_sphere_polydata, orig_surf_polydata, point_patchSize_dict_orig, output_prefix, 'orig')
    _, marginal_points_gyri, _ = find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, original_sulc_data)
    draw_3hinge_on_surf(orig_sphere_polydata, marginal_points_gyri, output_prefix + '_sphere_marginal.vtk')
    draw_3hinge_on_surf(orig_surf_polydata, marginal_points_gyri, output_prefix + '_surf_marginal.vtk')
    all_long_path_convex_points_list, convex_marginal_points_length_dict, island_convex_marginal_points_length_dict = find_convex_marginal_points_of_long_gyri(orig_sphere_polydata, final_point_lines_dict, point_neighbor_points_dict, point_connect_points_dict_gyri_parts, marginal_points_gyri, original_sulc_data, curv_data_delete_thicknessZero, point_patchSize_dict_orig, length_thres_of_long_gyri, nearest_skeleton_num, island_gyri_length_thres)
    draw_3hinge_on_surf(orig_sphere_polydata, all_long_path_convex_points_list, output_prefix + '_sphere_convex_marginal.vtk')
    draw_3hinge_on_surf(orig_surf_polydata, all_long_path_convex_points_list, output_prefix + '_surf_convex_marginal.vtk')
    long_path_convex_points_list = select_convex_endpoint_for_each_missing_gyri(all_long_path_convex_points_list, convex_marginal_points_length_dict, curv_data_delete_thicknessZero, point_connect_points_dict_gyri_parts, neighbor_missing_path_smallest_step)
    draw_3hinge_on_surf(orig_sphere_polydata, long_path_convex_points_list, output_prefix + '_sphere_convex_marginal_long.vtk')
    draw_3hinge_on_surf(orig_surf_polydata, long_path_convex_points_list, output_prefix + '_surf_convex_marginal_long.vtk')
    long_path_convex_points_list_sorted = sort_the_long_path_convex_point(long_path_convex_points_list, convex_marginal_points_length_dict)
    missing_connection_list, final_point_lines_dict = build_missing_connection_recurrently(long_path_convex_points_list_sorted, final_point_lines_dict, point_connect_points_dict_gyri_parts, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri)
    write_skelenton_by_connectionPair(orig_sphere_polydata, missing_connection_list, output_prefix + '_sphere_missing_connection.vtk')
    write_skelenton_by_connectionPair(orig_surf_polydata, missing_connection_list, output_prefix + '_surf_missing_connection.vtk')

    long_path_island_convex_candicate_points_list = list(island_convex_marginal_points_length_dict.keys())
    while long_path_island_convex_candicate_points_list:
        long_path_island_convex_candicate_points_list_sorted = sort_the_long_path_convex_point(long_path_island_convex_candicate_points_list,
                                                                                               island_convex_marginal_points_length_dict)
        current_longest_island_convex_point = long_path_island_convex_candicate_points_list_sorted[0]
        final_point_lines_dict, connection_dicts_list = build_main_skeleton_for_missing_island_gyri(current_longest_island_convex_point,
                                                                                                    final_point_lines_dict,
                                                                                                    point_connect_points_dict_gyri_parts,
                                                                                                    point_patchSize_dict_orig,
                                                                                                    curv_data_delete_thicknessZero,
                                                                                                    flat_threshold_for_convex_gyri)
        from_marginal_to_skelenton_father_sons_dict = connection_dicts_list[0]
        from_skelenton_to_marginal_son_fathers_dict = connection_dicts_list[1]
        island_convex_marginal_point_list = [point for point in from_skelenton_to_marginal_son_fathers_dict.keys()
                                             if point in island_convex_marginal_points_length_dict.keys()
                                             and point not in final_point_lines_dict.keys()]
        all_long_path_island_convex_points_list, convex_marginal_island_points_length_dict, _ = find_convex_marginal_points_of_long_gyri(
            orig_sphere_polydata, final_point_lines_dict, point_neighbor_points_dict,
            point_connect_points_dict_gyri_parts, island_convex_marginal_point_list, original_sulc_data,
            curv_data_delete_thicknessZero, point_patchSize_dict_orig, length_thres_of_long_gyri, nearest_skeleton_num,
            island_gyri_length_thres)
        long_path_island_convex_points_list = select_convex_endpoint_for_each_missing_gyri(all_long_path_island_convex_points_list,
                                                                                           convex_marginal_island_points_length_dict,
                                                                                           curv_data_delete_thicknessZero,
                                                                                           point_connect_points_dict_gyri_parts,
                                                                                           neighbor_missing_path_smallest_step)
        long_path_island_convex_points_list_sorted = sort_the_long_path_convex_point(long_path_island_convex_points_list,
                                                                                     convex_marginal_island_points_length_dict)
        _, final_point_lines_dict = build_missing_connection_recurrently(long_path_island_convex_points_list_sorted,
                                                                         final_point_lines_dict,
                                                                         point_connect_points_dict_gyri_parts,
                                                                         point_patchSize_dict_orig,
                                                                         curv_data_delete_thicknessZero,
                                                                         flat_threshold_for_convex_gyri)
        finished_island_convex_points_list = [point for point in island_convex_marginal_points_length_dict.keys()
                                              if point in from_marginal_to_skelenton_father_sons_dict.keys()
                                              or point in from_skelenton_to_marginal_son_fathers_dict.keys()]
        for point in finished_island_convex_points_list:
            long_path_island_convex_candicate_points_list.remove(point)

    return final_point_lines_dict


def find_the_max_sulc_point(points_list, original_sulc_data):
    max = -0
    for point in points_list:
        if original_sulc_data[point] < max:
            max = original_sulc_data[point]
            max_sulc_point = point
    return max_sulc_point


def read_thin_sphere_surf(file):
    thin_sphere_polydata = read_vtk_file(file)
    point_num = thin_sphere_polydata.GetNumberOfPoints()
    thin_sulc_data = list()
    for point in range(point_num):
        sulc = thin_sphere_polydata.GetPointData().GetArray('sulc_updated').GetTuple(point)[0]
        thin_sulc_data.append(sulc)
    thin_sulc_data = np.stack(thin_sulc_data)
    return thin_sphere_polydata, thin_sulc_data


def create_tree_connection_dict(connected_lines_list):
    point_lines_dict = defaultdict(set)
    for line in connected_lines_list:
        if line[0] not in point_lines_dict.keys():
            point_lines_dict[line[0]] = {line[1]}
        else:
            point_lines_dict[line[0]].add(line[1])
        if line[1] not in point_lines_dict.keys():
            point_lines_dict[line[1]] = {line[0]}
        else:
            point_lines_dict[line[1]].add(line[0])
    return point_lines_dict


def step_n_father(point, n, father_dict):
    point_father_list = list()
    current_father = father_dict[point]
    for step in range(n):
        point_father_list.append(current_father)
        current_father = father_dict[current_father]
    return point_father_list


def get_connection_degree_of_step_n_father(start_point, end_father, father_dict, point_lines_dict):
    connection_degrees = list()
    father = father_dict[start_point]
    while father != end_father:
        degree = len(point_lines_dict[father])
        connection_degrees.append(degree)
        father = father_dict[father]
    degree = len(point_lines_dict[father])
    connection_degrees.append(degree)
    return connection_degrees


def connect_break_in_circle(point_lines_dict, point_connect_points_dict_thin_gyri_parts, deleted_round1_points, father_dict):
    for point in point_lines_dict.keys():
        if len(point_lines_dict[point]) == 1:
            find_neighbor_endpoint = 0
            for neighbor in point_connect_points_dict_thin_gyri_parts[point]:
                if neighbor in point_lines_dict.keys() and len(point_lines_dict[neighbor]) == 1:
                    # pdb.set_trace()
                    if not list(set(step_n_father(point, 10, father_dict)) & set(step_n_father(neighbor, 10, father_dict))):
                        point_lines_dict[point].add(neighbor)
                        point_lines_dict[neighbor].add(point)
                        find_neighbor_endpoint = 1
                    else:
                        print('Case1-neighbor: connect two points belonging to the same branch:', point, neighbor)

            if find_neighbor_endpoint == 0:
                # if point == 109497:
                #     pdb.set_trace()
                for neighbor in point_connect_points_dict_thin_gyri_parts[point]:
                    if neighbor in deleted_round1_points:
                        # pdb.set_trace()
                        for second_neighbor in point_connect_points_dict_thin_gyri_parts[neighbor]:
                            if len(point_lines_dict[second_neighbor]) == 1 and second_neighbor != point:
                                if not list(set(step_n_father(point, 10, father_dict)) & set(step_n_father(second_neighbor, 10, father_dict))):
                                    point_lines_dict[point].add(neighbor)
                                    point_lines_dict[neighbor] = {point}
                                    point_lines_dict[neighbor].add(second_neighbor)
                                    point_lines_dict[second_neighbor].add(neighbor)
                                    find_neighbor_endpoint = 1
                                    break
                                else:
                                    print('Case2-deleted_round1: connect two points belonging to the same branch:', point, neighbor, second_neighbor)
                    if find_neighbor_endpoint:
                        break

            if find_neighbor_endpoint == 0:
                father = father_dict[point]
                for father_neighbor in point_connect_points_dict_thin_gyri_parts[father]:
                    if father_neighbor != point and len(point_lines_dict[father_neighbor]) == 1:
                        if not list(set(step_n_father(father, 10, father_dict)) & set(step_n_father(father_neighbor, 10, father_dict))):
                            point_lines_dict[father].add(father_neighbor)
                            point_lines_dict[father_neighbor].add(father)
                            point_lines_dict[father].remove(point)
                            point_lines_dict[point].remove(father)
                            find_neighbor_endpoint = 1
                            break
                        else:
                            print('Case3-father: connect two points belonging to the same branch:', father, father_neighbor)

            if find_neighbor_endpoint == 0:
                neighbor_list = [neighbor for neighbor in point_connect_points_dict_thin_gyri_parts[point]
                                 if neighbor != father_dict[point]]
                neighbor_second_neighbor_pairs_list = list()
                for neighbor in neighbor_list:
                    if len(point_lines_dict[neighbor]) > 0:
                        # pass
                        print('two endpoints neighbored!', point, neighbor)
                    else:
                        neighbor_second_neighbor_pairs = [[neighbor, second_neighbor] for second_neighbor in
                                                          point_connect_points_dict_thin_gyri_parts[neighbor]
                                                          if second_neighbor in point_lines_dict.keys()
                                                          and len(point_lines_dict[second_neighbor]) == 1
                                                          and second_neighbor not in [point, father_dict[point], father_dict[father_dict[point]]]]
                        neighbor_second_neighbor_pairs_list = neighbor_second_neighbor_pairs_list + neighbor_second_neighbor_pairs
                for neighbor_second_neighbor_pair in neighbor_second_neighbor_pairs_list:
                    neighbor = neighbor_second_neighbor_pair[0]
                    second_neighbor = neighbor_second_neighbor_pair[1]
                    if not list(set(step_n_father(point, 10, father_dict)) & set(step_n_father(second_neighbor, 10, father_dict))):
                        point_lines_dict[point].add(neighbor)
                        point_lines_dict[neighbor] = {point}
                        point_lines_dict[neighbor].add(second_neighbor)
                        point_lines_dict[second_neighbor].add(neighbor)
                        find_neighbor_endpoint = 1
                        break
                    else:
                        print('Case4-round2:connect two points belonging to the same branch:', point, neighbor, second_neighbor)

            if find_neighbor_endpoint == 0:
                first_deleted_round1_neighbor_list = [neighbor for neighbor in point_connect_points_dict_thin_gyri_parts[point]
                                                      if neighbor in deleted_round1_points]
                if len(first_deleted_round1_neighbor_list):
                    for first_deleted_round1_neighbor in first_deleted_round1_neighbor_list:
                        second_deleted_round1_neighbor_list = [second_neighbor for second_neighbor in
                                                               point_connect_points_dict_thin_gyri_parts[first_deleted_round1_neighbor]
                                                               if second_neighbor in deleted_round1_points
                                                               and second_neighbor not in first_deleted_round1_neighbor_list]
                        for second_deleted_round1_neighbor in second_deleted_round1_neighbor_list:
                            for third_neighbor in point_connect_points_dict_thin_gyri_parts[second_deleted_round1_neighbor]:
                                if len(point_lines_dict[third_neighbor]) == 1:
                                    if not list(set(step_n_father(point, 10, father_dict)) & set(step_n_father(third_neighbor, 10, father_dict))):
                                        point_lines_dict[point].add(first_deleted_round1_neighbor)
                                        point_lines_dict[first_deleted_round1_neighbor].add(point)
                                        point_lines_dict[first_deleted_round1_neighbor].add(second_deleted_round1_neighbor)
                                        point_lines_dict[second_deleted_round1_neighbor].add(first_deleted_round1_neighbor)
                                        point_lines_dict[second_deleted_round1_neighbor].add(third_neighbor)
                                        point_lines_dict[third_neighbor].add(second_deleted_round1_neighbor)
                                        find_neighbor_endpoint = 1
                                        break
                                    else:
                                        print('Case5-deleted_round1: connect two points belonging to the same branch:',
                                              point, first_deleted_round1_neighbor, second_deleted_round1_neighbor, third_neighbor)
                            if find_neighbor_endpoint:
                                break
                        if find_neighbor_endpoint:
                            break

            if find_neighbor_endpoint == 0:
                print('remaining endpoint:', point)
    return point_lines_dict


def clear_empty_point_in_dict(point_lines_dict):
    clear_point_lines_dict = defaultdict(set)
    for point in point_lines_dict.keys():
        if len(point_lines_dict[point]):
            clear_point_lines_dict[point] = point_lines_dict[point]
    return clear_point_lines_dict


def trim_allRound1_long(point_lines_dict, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, father_dict, inner_gyri_points):
    '''delete round-1 short branches'''
    deleted_round1_points = list()
    for point in point_lines_dict.keys():
        connecting_points = list(point_lines_dict[point])
        long_edge_num = 0
        short_edge_max_patchsize = 0
        for connecting_point in connecting_points:
            if len(point_lines_dict[connecting_point]) > 1:
                long_edge_num += 1
            elif len(point_lines_dict[connecting_point]) == 1:
                if point_patchSize_dict_updated[connecting_point] > short_edge_max_patchsize:
                    short_edge_max_patchsize = point_patchSize_dict_updated[connecting_point]
                    short_edge_candidate = connecting_point
        current_deleted_round1_points_list = list()
        if long_edge_num < len(connecting_points):
            for connecting_point in connecting_points:
                if len(point_lines_dict[connecting_point]) == 1 and connecting_point != short_edge_candidate:
                    point_lines_dict[point].remove(connecting_point)
                    point_lines_dict[connecting_point].remove(point)
                    current_deleted_round1_points_list.append(connecting_point)
            if long_edge_num > 1:
                point_lines_dict[point].remove(short_edge_candidate)
                point_lines_dict[short_edge_candidate].remove(point)
                current_deleted_round1_points_list.append(short_edge_candidate)
        if long_edge_num == 1:
            deleted_round1_points = list(set(deleted_round1_points + current_deleted_round1_points_list))

    '''delete long branches'''
    branch_endpoint_list = list()
    for endpoint in point_lines_dict.keys():
        if len(point_lines_dict[endpoint]) == 1:
            branch_endpoint_list.append(endpoint)
    for endpoint in branch_endpoint_list:
        # if endpoint in [45287, 99484, 119924]:
        #     pdb.set_trace()
        step_n_father_list = step_n_father(endpoint, 6, father_dict)
        first_round_referencePoint_list_of_endpoint = [referencePoint_of_endpoint for referencePoint_of_endpoint in
                                                       point_connect_points_dict_thin_gyri_parts[endpoint]
                                                       if referencePoint_of_endpoint in point_lines_dict.keys()
                                                       and len(point_lines_dict[referencePoint_of_endpoint]) >= 2
                                                       and (referencePoint_of_endpoint not in step_n_father_list
                                                            or (referencePoint_of_endpoint in step_n_father_list
                                                                and max(get_connection_degree_of_step_n_father(endpoint, referencePoint_of_endpoint, father_dict, point_lines_dict)) > 2))]
        if len(first_round_referencePoint_list_of_endpoint) > 0:
            referencePoint_list_of_endpoint = list(set(first_round_referencePoint_list_of_endpoint))
        else:
            inner_neighbor_list = [inner_neighbor for inner_neighbor in point_connect_points_dict_thin_gyri_parts[endpoint]
                                   if inner_neighbor in inner_gyri_points]
            second_round_referencePoint_list_of_endpoint = list()
            for inner_neighbor in inner_neighbor_list:
                referencePoint_list = [referencePoint for referencePoint in
                                       point_connect_points_dict_thin_gyri_parts[inner_neighbor]
                                       if referencePoint in point_lines_dict.keys()
                                       and len(point_lines_dict[referencePoint]) >= 2
                                       and (referencePoint not in step_n_father_list
                                            or (referencePoint in step_n_father_list
                                                and max(get_connection_degree_of_step_n_father(endpoint, referencePoint, father_dict, point_lines_dict)) > 2))]
                second_round_referencePoint_list_of_endpoint = second_round_referencePoint_list_of_endpoint + referencePoint_list
            referencePoint_list_of_endpoint = list(set(second_round_referencePoint_list_of_endpoint))

        same_branch = 0
        for referencePoint_of_endpoint in referencePoint_list_of_endpoint:
            if father_dict[referencePoint_of_endpoint] in point_connect_points_dict_thin_gyri_parts[father_dict[endpoint]] or father_dict[referencePoint_of_endpoint] in point_connect_points_dict_thin_gyri_parts[father_dict[father_dict[endpoint]]]:
                same_branch = 1
                break
            else:
                endpoint_father_list = step_n_father(endpoint, 12, father_dict)
                referencePoint_father_list = step_n_father(referencePoint_of_endpoint, 12, father_dict)
                if list(set(endpoint_father_list) & set(referencePoint_father_list)):
                    same_branch = 1
                    break

        if same_branch:
            current_endpoint = endpoint
            while len(point_lines_dict[current_endpoint]) == 1:
                # pdb.set_trace()
                next_endpoint = list(point_lines_dict[current_endpoint])[0]
                point_lines_dict[current_endpoint].remove(next_endpoint)
                point_lines_dict[next_endpoint].remove(current_endpoint)
                current_endpoint = next_endpoint

    return point_lines_dict, deleted_round1_points


def create_connectingPair(point_lines_dict):
    final_connection_list = list()
    for point in point_lines_dict.keys():
        for connecting_point in point_lines_dict[point]:
            if [point, connecting_point] not in final_connection_list and [connecting_point, point] not in final_connection_list:
                final_connection_list.append([point, connecting_point])
    return final_connection_list


def create_3hinge(final_point_lines_dict):
    hinge3_list = list()
    for point in final_point_lines_dict.keys():
        if len(final_point_lines_dict[point]) >= 3:
            hinge3_list.append(point)
    return hinge3_list


def draw_3hinge_on_surf(surf_polydata, hinge3_list, output_3hinge_vertex):
    points_new = vtk.vtkPoints()
    vertices_new = vtk.vtkCellArray()
    vertex_num = 0
    for point in hinge3_list:
        coordinate = surf_polydata.GetPoints().GetPoint(point)
        points_new.InsertNextPoint(coordinate)
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, vertex_num)
        vertices_new.InsertNextCell(vertex)
        vertex_num += 1
    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points_new)
    polygonPolyData.SetVerts(vertices_new)
    polygonPolyData.Modified()

    if vtk.VTK_MAJOR_VERSION <= 5:
        polygonPolyData = polygonPolyData.GetProducerPort()
    else:
        polygonPolyData = polygonPolyData

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polygonPolyData)
    writer.SetFileName(output_3hinge_vertex)
    writer.Write()


def create_shortest_path(adj, start, end):
    # pdb.set_trace()
    # for i in range(adj.shape[0]):
    #     print(adj[i, :].min(), adj[:, i].min())
    G = nx.Graph(adj)
    r_list_numpy_index = nx.dijkstra_path(G, source=start, target=end)
    length = nx.dijkstra_path_length(G, source=start, target=end)
    return r_list_numpy_index, length


def draw_patchSize_colorful_file(sphere_polydata, surf_polydata, point_patchSize_dict, output_prefix, type):
    patchSize_data = list()
    for point in range(sphere_polydata.GetNumberOfPoints()):
        patchSize_data.append(point_patchSize_dict[point])
    write_featured_sphere_from_variable_single(sphere_polydata, type, patchSize_data, output_prefix + '_sphere_patchSize_' + type + '.vtk')
    write_featured_sphere_from_variable_single(surf_polydata, type, patchSize_data, output_prefix + '_surf_patchSize_' + type + '.vtk')


def main(args):
    root = args.root_dir
    subject_index_start = args.subject_list_start_id
    # print(subject_index_start)
    subject_index_end = args.subject_list_end_id
    # subjects_list = [str(subject) for subject in os.listdir(root) if not subject.startswith('.')]
    # subjects_list.sort()
    # current_subject_list = subjects_list[subject_index_start:subject_index_end]
    current_subject_list = [subject_index_start]
    print(current_subject_list)

    sphere_list = args.sphere_list
    for subject in current_subject_list:
        # print(subject)
        subject = str(subject)

        out_dir = root + '/' + args.out_dir

        if not os.path.exists(out_dir):
            # shutil.rmtree(out_dir)
            os.mkdir(out_dir)

        for sphere in sphere_list:
            result_file = f'{sphere}_surf_3hinge_vertex.vtk'
            result_file_path = os.path.join(out_dir, result_file)
            #print(result_file_path)

            if os.path.exists(result_file_path):
                continue

            sphere_file = root + '/' + args.input_dir + '/' + 'surf' + '/' +sphere + args.sphere_file
            surf_file = root + '/' + sphere + args.surf_file
            curv_file = root + '/' + args.input_dir + '/' + 'surf' + '/' +sphere + args.curv_file
            sulc_file = root + '/' + args.input_dir + '/' + 'surf' + '/' +sphere + args.sulc_file
            thickness_file = root +'/' + args.input_dir + '/' + 'surf' + '/' + sphere + args.thickness_file

            output_prefix = out_dir + '/' + sphere
            feature_file_dict = {'sulc': sulc_file, 'curv': curv_file, 'thickness': thickness_file}

            sphere_polydata = read_vtk_file(sphere_file)
            surf_polydata = read_vtk_file(surf_file)

            # featured_sphere(sphere_polydata, feature_file_dict, output_prefix + '_sphere_features.vtk')
            # featured_sphere(surf_polydata, feature_file_dict, output_prefix + '_surf_features.vtk')
            featured_sphere(sphere_polydata, feature_file_dict, output_prefix + '_sphere_features_GDM.vtk')
            featured_sphere(surf_polydata, feature_file_dict, output_prefix + '_surf_features_GDM.vtk')

            print('create points connection dict:\t' + time.asctime(time.localtime(time.time())))
            point_neighbor_points_dict = get_connect_points(sphere_polydata)
            updated_sulc_data, original_sulc_data, curv_data_delete_thicknessZero = initialize_sulc_data2(sphere_polydata,
                                                                                                         surf_polydata,
                                                                                                         feature_file_dict,
                                                                                                         point_neighbor_points_dict,
                                                                                                         args.inner_sulci_curv_thres,
                                                                                                         args.inner_sulci_round_thres,
                                                                                                         args.outer_gyri_curv_thres,
                                                                                                         args.inner_sulci_neighbor_curv_thres,
                                                                                                         output_prefix,
                                                                                                         sulc_threshold=0.3)

            print('calculate patchsize of gyri part:\t' + time.asctime(time.localtime(time.time())))
            point_patchSize_dict_updated = find_the_patchSize_of_gyri_point(updated_sulc_data, point_neighbor_points_dict)
            print('draw patchSize colorful sphere:\t' + time.asctime(time.localtime(time.time())))
            draw_patchSize_colorful_file(sphere_polydata, surf_polydata, point_patchSize_dict_updated, output_prefix, 'updated')
            
            thin_sulc_data = write_thin_gyri_on_sphere_point_marginal_sulc_curv(sphere_polydata, surf_polydata,
                                                                                point_neighbor_points_dict,
                                                                                updated_sulc_data,
                                                                                curv_data_delete_thicknessZero,
                                                                                args.expend_curv_step_size,
                                                                                output_prefix)
            print('create thin gyri connection dict \t' + time.asctime(time.localtime(time.time())))
            point_connect_points_dict_thin_gyri_parts = get_connect_points_gyri_part(sphere_polydata, thin_sulc_data)
            
            connected_lines_list, father_dict = create_tree(sphere_polydata, surf_polydata, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, thin_sulc_data, output_prefix)
            
            find_skelenton(sphere_polydata, surf_polydata, point_patchSize_dict_updated, curv_data_delete_thicknessZero,
                           original_sulc_data, thin_sulc_data, point_neighbor_points_dict, point_connect_points_dict_thin_gyri_parts,
                           connected_lines_list, father_dict, args.length_thres_of_long_gyri, args.neighbor_missing_path_smallest_step,
                           args.flat_threshold_for_convex_gyri, args.nearest_skeleton_num, args.island_gyri_length_thres, output_prefix)

            skeleton_polydata = read_vtk_file(output_prefix + '_sphere_skelenton_connect_break_allpoints.vtk')
            find_skelenton_missing2(sphere_polydata, surf_polydata, skeleton_polydata, curv_data_delete_thicknessZero,
                                   original_sulc_data,
                                   args.length_thres_of_long_gyri, args.neighbor_missing_path_smallest_step,
                                   args.flat_threshold_for_convex_gyri, args.nearest_skeleton_num, args.island_gyri_length_thres, output_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GyralNet creation by expending algorithm")
    parser.add_argument('-root_dir', '--root_dir', type=str, default='./', help='root for input data')

    parser.add_argument('-subject_list_start_id', '--subject_list_start_id', type=int, default=0, help='subjects list start and end ids')
    parser.add_argument('-subject_list_end_id', '--subject_list_end_id', type=int, default=-1, help='subjects list start and end ids')

    parser.add_argument('-input_dir', '--input_dir', type=str, default='_recon', help='input dir within each subject')
    parser.add_argument('-out_dir', '--out_dir', type=str, default='gyralnet_island_GDM0.3', help='out dir within each subject')
    parser.add_argument('-sphere_list', '--sphere_list', type=list, default=['lh', 'rh'], help='spheres')
    parser.add_argument('-sphere_file', '--sphere_file', type=str, default='.withGrad.32k_fs_LR.Sphere.vtk', help='sphere_file name')

    parser.add_argument('-surf_file', '--surf_file', type=str, default='.withGrad.32k_fs_LR.Inner.vtk', help='surf_file name')
    parser.add_argument('-curv_file', '--curv_file', type=str, default='.flip.curv', help='curv_file name')
    parser.add_argument('-sulc_file', '--sulc_file', type=str, default='.grad.sulc', help='sulc_file name')
    parser.add_argument('-thickness_file', '--thickness_file', type=str, default='.thickness', help='thickness_file name')

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

