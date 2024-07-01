import vtk
import nibabel.freesurfer.io as io
from collections import defaultdict
import numpy as np
import os
import shutil
import time
import networkx as nx
import argparse
from utils import featured_sphere


def read_vtk_file(vtk_file):
    """
    读取VTK文件并返回PolyData对象。

    参数:
        vtk_file (str): VTK文件路径。

    返回:
        vtk.vtkPolyData: 包含VTK数据的PolyData对象。
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    return reader.GetOutput()


def get_connect_points(surf_polydata):
    """
    获取顶点之间连通性的字典。

    参数:
        surf_polydata (vtk.vtkPolyData): 表面PolyData对象。

    返回:
        dict: 顶点之间连通性的字典，以顶点索引为键，邻接顶点索引列表为值。
    """
    point_connect_points_dict = defaultdict(list)
    cell_num = surf_polydata.GetNumberOfCells()
    CellArray = surf_polydata.GetPolys()
    Polygons = CellArray.GetData()

    for i in range(cell_num):
        triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
        for j in range(3):
            if triangle[(j + 1) % 3] not in point_connect_points_dict[triangle[j]]:
                point_connect_points_dict[triangle[j]].append(triangle[(j + 1) % 3])
            if triangle[j] not in point_connect_points_dict[triangle[(j + 1) % 3]]:
                point_connect_points_dict[triangle[(j + 1) % 3]].append(triangle[j])
    return point_connect_points_dict


def get_connect_points_gyri_part(surf_polydata, sulc_data):
    """
    获取只包括sulc值小于0的顶点之间连通性的字典，即Gyri部分。

    参数:
        surf_polydata (vtk.vtkPolyData): 表面PolyData对象。
        sulc_data (numpy.ndarray): Sulc数据数组。

    返回:
        dict: 只包括Gyri部分的顶点之间连通性的字典。
    """
    point_connect_points_dict_gyri_parts = defaultdict(list)
    cell_num = surf_polydata.GetNumberOfCells()
    CellArray = surf_polydata.GetPolys()
    Polygons = CellArray.GetData()

    for i in range(cell_num):
        triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
        for j in range(3):
            k = (j + 1) % 3
            if sulc_data[triangle[j]] < 0 and sulc_data[triangle[k]] < 0:
                if triangle[k] not in point_connect_points_dict_gyri_parts[triangle[j]]:
                    point_connect_points_dict_gyri_parts[triangle[j]].append(triangle[k])
                if triangle[j] not in point_connect_points_dict_gyri_parts[triangle[k]]:
                    point_connect_points_dict_gyri_parts[triangle[k]].append(triangle[j])
    return point_connect_points_dict_gyri_parts


def delete_isolated_point(point_num, point_neighbor_points_dict, sulc_data):
    """
    删除孤立的点，即周围的所有点都与其相反的sulc值。

    参数:
        point_num (int): 点的数量。
        point_neighbor_points_dict (dict): 顶点之间连通性的字典。
        sulc_data (numpy.ndarray): Sulc数据数组。

    返回:
        numpy.ndarray: 更新后的Sulc数据数组。
    """
    marginal_points, marginal_points_gyri, marginal_points_sulc = find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, sulc_data)

    for point in range(point_num):
        sulc = sulc_data[point]
        neighbor_sulcs = sulc_data[point_neighbor_points_dict[point]]

        if sulc > 0 and np.all(neighbor_sulcs < 0):
            sulc_data[point] = -1
        elif sulc < 0 and np.all(neighbor_sulcs > 0):
            sulc_data[point] = 1
        else:
            if sulc > 0:
                first_neighbor_list = [first_neighbor for first_neighbor in point_neighbor_points_dict[point] if first_neighbor not in marginal_points_gyri]
            elif sulc < 0:
                first_neighbor_list = [first_neighbor for first_neighbor in point_neighbor_points_dict[point] if first_neighbor not in marginal_points_sulc]

            second_neighbor_list = list()
            for first_neighbor in first_neighbor_list:
                second_neighbors = [second_neighbor for second_neighbor in point_neighbor_points_dict[first_neighbor] if second_neighbor not in [point] + first_neighbor_list]
                second_neighbor_list += second_neighbors
            second_neighbor_list = list(set(second_neighbor_list))

            if sulc > 0 and np.all(sulc_data[second_neighbor_list] < 0):
                sulc_data[[point] + first_neighbor_list] = -1
            elif sulc < 0 and np.all(sulc_data[second_neighbor_list] > 0):
                sulc_data[[point] + first_neighbor_list] = 1

    return sulc_data


def delete_isolated_point1(point_num, point_neighbor_points_dict, sulc_data):
    """
    删除孤立的点和一轮孤立的区域。

    参数:
        point_num (int): 点的数量。
        point_neighbor_points_dict (dict): 顶点之间连通性的字典。
        sulc_data (numpy.ndarray): Sulc数据数组。

    返回:
        numpy.ndarray: 更新后的Sulc数据数组。
    """
    for point in range(point_num):
        sulc = sulc_data[point]
        neighbor_sulcs = sulc_data[point_neighbor_points_dict[point]]

        if sulc > 0 and np.all(neighbor_sulcs < 0):
            sulc_data[point] = -1
        elif sulc < 0 and np.all(neighbor_sulcs > 0):
            sulc_data[point] = 1
        else:
            first_neighbor_list = point_neighbor_points_dict[point]
            second_neighbor_list = list()
            for first_neighbor in first_neighbor_list:
                neighbors_of_first_round = point_neighbor_points_dict[first_neighbor]
                second_neighbor_list += [neighbor for neighbor in neighbors_of_first_round if neighbor not in [point] + first_neighbor_list]
            second_neighbor_list = list(set(second_neighbor_list))
            if sulc > 0 and np.all(sulc_data[second_neighbor_list] < 0):
                sulc_data[[point] + first_neighbor_list] = -1
            elif sulc < 0 and np.all(sulc_data[second_neighbor_list] > 0):
                sulc_data[[point] + first_neighbor_list] = 1

    return sulc_data


def find_marginal_point(points_list, point_neighbor_points_dict, sulc_data):
    """
    找到边缘点，即邻接点中sulc值不一致的点。

    参数:
        points_list (list): 顶点索引列表。
        point_neighbor_points_dict (dict): 顶点之间连通性的字典。
        sulc_data (numpy.ndarray): Sulc数据数组。

    返回:
        tuple: (所有边缘点的列表, Gyri部分的边缘点列表, Sulci部分的边缘点列表)
    """
    marginal_points = list()
    marginal_points_gyri = list()
    marginal_points_sulc = list()

    for point in points_list:
        neighbor_points = point_neighbor_points_dict[point]
        neighbor_sulcs = sulc_data[neighbor_points]
        if np.any(neighbor_sulcs < 0) and np.any(neighbor_sulcs > 0):
            marginal_points.append(point)
            if sulc_data[point] < 0:
                marginal_points_gyri.append(point)
            else:
                marginal_points_sulc.append(point)

    return marginal_points, marginal_points_gyri, marginal_points_sulc


def featured_sphere(orig_sphere_polydata, feature_file_dict, output):
    """
    创建包含特征数据的sphere VTK文件。

    参数:
        orig_sphere_polydata (vtk.vtkPolyData): 原始sphere PolyData对象。
        feature_file_dict (dict): 包含特征名和特征文件路径的字典。
        output (str): 输出文件路径。
    """
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    cell_num = orig_sphere_polydata.GetNumberOfCells()

    with open(output, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("mesh surface\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write("POINTS " + str(point_num) + " float\n")

        for point in range(point_num):
            coordinate = orig_sphere_polydata.GetPoints().GetPoint(point)
            f.write(f"{coordinate[0]} {coordinate[1]} {coordinate[2]}\n")

        f.write("POLYGONS " + str(cell_num) + " " + str(4 * cell_num) + '\n')
        CellArray = orig_sphere_polydata.GetPolys()
        Polygons = CellArray.GetData()
        for i in range(cell_num):
            triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
            f.write(f"3 {triangle[0]} {triangle[1]} {triangle[2]}\n")

        f.write("POINT_DATA " + str(point_num) + '\n')
        scale_dict = {'sulc': 0.0, 'curv': 0.0, 'thickness': 1e-10}
        for feature_name, scale in scale_dict.items():
            feature_file = feature_file_dict[feature_name]
            features = io.read_morph_data(feature_file)

            f.write(f"SCALARS {feature_name}_binary float\n")
            f.write(f"LOOKUP_TABLE {feature_name}_binary\n")
            for feature in features:
                if feature >= scale:
                    f.write("1\n")
                elif feature < -scale:
                    f.write("-1\n")
                else:
                    f.write("0\n")

            f.write(f"SCALARS {feature_name} float\n")
            f.write(f"LOOKUP_TABLE {feature_name}\n")
            for feature in features:
                f.write(f"{feature}\n")


def write_featured_sphere_from_variable(orig_sphere_polydata, feature_name_variable_dict, output):
    """
    创建包含特征数据的sphere VTK文件，从变量中读取特征数据。

    参数:
        orig_sphere_polydata (vtk.vtkPolyData): 原始sphere PolyData对象。
        feature_name_variable_dict (dict): 包含特征名和特征数据数组的字典。
        output (str): 输出文件路径。
    """
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    cell_num = orig_sphere_polydata.GetNumberOfCells()

    with open(output, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("mesh surface\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write("POINTS " + str(point_num) + " float\n")

        for point in range(point_num):
            coordinate = orig_sphere_polydata.GetPoints().GetPoint(point)
            f.write(f"{coordinate[0]} {coordinate[1]} {coordinate[2]}\n")

        f.write("POLYGONS " + str(cell_num) + " " + str(4 * cell_num) + '\n')
        CellArray = orig_sphere_polydata.GetPolys()
        Polygons = CellArray.GetData()
        for i in range(cell_num):
            triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
            f.write(f"3 {triangle[0]} {triangle[1]} {triangle[2]}\n")

        f.write("POINT_DATA " + str(point_num) + '\n')
        for feature_name, features in feature_name_variable_dict.items():
            f.write(f"SCALARS {feature_name} float\n")
            f.write(f"LOOKUP_TABLE {feature_name}\n")
            for feature in features:
                f.write(f"{feature}\n")


def write_featured_sphere_from_variable_single(orig_sphere_polydata, feature_name, feature_variable, output):
    """
    创建包含单个特征数据的sphere VTK文件，从变量中读取特征数据。

    参数:
        orig_sphere_polydata (vtk.vtkPolyData): 原始sphere PolyData对象。
        feature_name (str): 特征名。
        feature_variable (numpy.ndarray): 特征数据数组。
        output (str): 输出文件路径。
    """
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    cell_num = orig_sphere_polydata.GetNumberOfCells()

    with open(output, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("mesh surface\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write("POINTS " + str(point_num) + " float\n")

        for point in range(point_num):
            coordinate = orig_sphere_polydata.GetPoints().GetPoint(point)
            f.write(f"{coordinate[0]} {coordinate[1]} {coordinate[2]}\n")

        f.write("POLYGONS " + str(cell_num) + " " + str(4 * cell_num) + '\n')
        CellArray = orig_sphere_polydata.GetPolys()
        Polygons = CellArray.GetData()
        for i in range(cell_num):
            ename = [Polygons.GetValue(idx) for idx in range(i * 4 + 1, i * 4 + 4)]
            f.write(f"3 {ename[0]} {ename[1]} {ename[2]}\n")

        f.write("POINT_DATA " + str(point_num) + '\n')
        f.write(f"SCALARS {feature_name} float\n")
        f.write(f"LOOKUP_TABLE {feature_name}\n")
        for feature in feature_variable:
            f.write(f"{feature}\n")


def find_round_n_neighbor(point, n, point_neighbor_points_dict):
    """
    找到某个点在n轮内的所有邻接点。

    参数:
        point (int): 点索引。
        n (int): 搜索轮数。
        point_neighbor_points_dict (dict): 顶点之间连通性的字典。

    返回:
        tuple: 包含所有邻接点和当前外层点的列表（元组的格式是 (neighbor_list, current_outer_points_list)）。
    """
    neighbor_list = []
    current_outer_points_list = [point]

    while n > 0:
        neighbor_list = list(set(neighbor_list + current_outer_points_list))
        next_outer_points_list = []

        for p in current_outer_points_list:
            neighbors = [neighbor for neighbor in point_neighbor_points_dict[p] if neighbor not in neighbor_list]
            next_outer_points_list += neighbors

        next_outer_points_list = list(set(next_outer_points_list))
        current_outer_points_list = next_outer_points_list
        n -= 1

    neighbor_list = list(set(neighbor_list + current_outer_points_list))
    return neighbor_list, current_outer_points_list


def find_missing_gyri_by_sulc_and_curv(sulc_data, curv_data, outer_gyri_curv_thres):
    """
    通过Sulc和曲率数据找到缺失的Gyri点。

    参数:
        sulc_data (numpy.ndarray): Sulc数据数组。
        curv_data (numpy.ndarray): 曲率数据数组。
        outer_gyri_curv_thres (float): 外部Gyri曲率阈值。

    返回:
        list: 缺失的Gyri点的索引列表。
    """
    missing_gyri_point_list = [point for point in range(sulc_data.shape[0]) if sulc_data[point] >= 0 and curv_data[point] < outer_gyri_curv_thres]
    return missing_gyri_point_list


def find_inner_sulci_by_sulc_and_curv(sulc_data, curv_data, point_neighbor_points_dict, inner_sulci_curv_thres, inner_sulci_round_thres, inner_sulci_neighbor_curv_thres):
    """
    通过Sulc和曲率数据找到内部Sulci点。

    参数:
        sulc_data (numpy.ndarray): Sulc数据数组。
        curv_data (numpy.ndarray): 曲率数据数组。
        point_neighbor_points_dict (dict): 顶点之间连通性的字典。
        inner_sulci_curv_thres (float): 内部Sulci曲率阈值。
        inner_sulci_round_thres (int): 内部Sulci轮数阈值。
        inner_sulci_neighbor_curv_thres (float): 内部Sulci邻接曲率阈值。

    返回:
        list: 内部Sulci点的索引列表。
    """
    inner_sulci_point_list = []

    for point in range(sulc_data.shape[0]):
        if sulc_data[point] < 0 and curv_data[point] > inner_sulci_curv_thres:
            neighbor_collection, round_n_neighbors = find_round_n_neighbor(point, inner_sulci_round_thres, point_neighbor_points_dict)

            if np.sum(curv_data[round_n_neighbors] > inner_sulci_neighbor_curv_thres) > 0:
                for neighbor in neighbor_collection:
                    if curv_data[neighbor] > 0:
                        inner_sulci_point_list.append(neighbor)

    return inner_sulci_point_list


def update_sulc_data(original_sulc_data, missing_gyri_point_list, inner_sulci_point_list):
    """
    更新Sulc数据，将缺失的Gyri点和内部Sulci点标记出来。

    参数:
        original_sulc_data (numpy.ndarray): 原始Sulc数据数组。
        missing_gyri_point_list (list): 缺失Gyri点的索引列表。
        inner_sulci_point_list (list): 内部Sulci点的索引列表。

    返回:
        numpy.ndarray: 更新后的Sulc数据数组。
    """
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
    """
    去除厚度数据的噪声，将孤立的0厚度点设置为1。

    参数:
        point_num (int): 点的数量。
        point_neighbor_points_dict (dict): 顶点之间连通性的字典。
        thickness_data_raw (numpy.ndarray): 原始厚度数据数组。

    返回:
        numpy.ndarray: 去噪后的厚度数据数组。
    """
    thickness_data = thickness_data_raw.copy()

    for point in range(point_num):
        if thickness_data[point] == 0:
            step_n = 1
            noise = False

            while not noise and step_n <= 5:
                neighbor_list, outer_points_list = find_round_n_neighbor(point, step_n, point_neighbor_points_dict)

                if np.sum(thickness_data[outer_points_list] > 0) == len(outer_points_list):
                    thickness_data[neighbor_list] = 1
                    noise = True
                else:
                    step_n += 1

    return thickness_data


def initialize_sulc_data(orig_sphere_polydata, orig_surf_polydata, feature_file_dict, point_neighbor_points_dict, inner_sulci_curv_thres, inner_sulci_round_thres, outer_gyri_curv_thres, inner_sulci_neighbor_curv_thres, output_prefix):
    """
    初始化Sulc数据，进行去噪和更新。

    参数:
        orig_sphere_polydata (vtk.vtkPolyData): 原始sphere PolyData对象。
        orig_surf_polydata (vtk.vtkPolyData): 原始surf PolyData对象。
        feature_file_dict (dict): 包含特征名和特征文件路径的字典。
        point_neighbor_points_dict (dict): 顶点之间连通性的字典。
        inner_sulci_curv_thres (float): 内部Sulci曲率阈值。
        inner_sulci_round_thres (int): 内部Sulci轮数阈值。
        outer_gyri_curv_thres (float): 外部Gyri曲率阈值。
        inner_sulci_neighbor_curv_thres (float): 内部Sulci邻接曲率阈值。
        output_prefix (str): 输出文件前缀。

    返回:
        tuple: (更新后的Sulc数据数组, 原始Sulc数据数组, 删除厚度0的曲率数据数组)
    """
    point_num = orig_sphere_polydata.GetNumberOfPoints()

    print('更新Sulc值根据曲率：' + time.asctime(time.localtime(time.time())))

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

    print('绘制更新后的Sulc彩色球体：' + time.asctime(time.localtime(time.time())))
    feature_name_variable_dict = {
        'thickness_data_raw': thickness_data_raw,
        'thickness_de_noise': thickness_data,
        'sulc_data_raw': sulc_data_raw,
        'sulc_data_delete_thicknessZero': sulc_data_delete_thicknessZero,
        'sulc_data_binary': sulc_data_binary,
        'sulc_data_de_isolation': sulc_data,
        'curv_data_raw': curv_data_raw,
        'curv_data_delete_thicknessZero': curv_data_delete_thicknessZero,
        'updated_sulc_data': updated_sulc_data
    }

    write_featured_sphere_from_variable(orig_sphere_polydata, feature_name_variable_dict, output_prefix + '_sphere_feature_updated.vtk')
    write_featured_sphere_from_variable(orig_surf_polydata, feature_name_variable_dict, output_prefix + '_surf_feature_updated.vtk')

    return updated_sulc_data, sulc_data, curv_data_delete_thicknessZero


def write_thin_gyri_on_sphere_point_marginal_sulc_curv(orig_sphere_polydata, orig_surf_polydata, point_neighbor_points_dict, updated_sulc_data, curv_data_delete_thicknessZero, expend_curv_step_size, output_prefix):
    """
    在球面上绘制修整后的Gyri轨迹。

    参数:
        orig_sphere_polydata (vtk.vtkPolyData): 原始sphere PolyData对象。
        orig_surf_polydata (vtk.vtkPolyData): 原始surf PolyData对象。
        point_neighbor_points_dict (dict): 顶点之间连通性的字典。
        updated_sulc_data (numpy.ndarray): 更新后的Sulc数据数组。
        curv_data_delete_thicknessZero (numpy.ndarray): 删除厚度0的曲率数据数组。
        expend_curv_step_size (float): 扩展过程中曲率步长大小。
        output_prefix (str): 输出文件前缀。

    返回:
        numpy.ndarray: 更新后的Sulc数据数组。
    """
    print('================= 开始侵蚀: ' + time.asctime(time.localtime(time.time())) + ' =================')
    point_num = orig_sphere_polydata.GetNumberOfPoints()
    cell_num = orig_sphere_polydata.GetNumberOfCells()
    CellArray = orig_sphere_polydata.GetPolys()
    Polygons = CellArray.GetData()

    print('创建三角形连接：' + time.asctime(time.localtime(time.time())))
    triangle_collection = [set([Polygons.GetValue(j) for j in range(cell * 4 + 1, cell * 4 + 4)]) for cell in range(cell_num)]

    print('找到初始边缘点：' + time.asctime(time.localtime(time.time())))
    marginal_points, marginal_points_gyri, marginal_points_sulc = find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, updated_sulc_data)

    min_curv = curv_data_delete_thicknessZero.min()

    print('开始扩展：' + time.asctime(time.localtime(time.time())))
    round_num = 0
    current_curv = 0
    curv_step = expend_curv_step_size

    while current_curv >= min_curv:
        current_curv -= curv_step
        finish = 1
        print('\n================ 当前曲率: ', current_curv, '====================')

        while finish > 0:
            round_num += 1
            print('第', round_num, '轮：' + time.asctime(time.localtime(time.time())))
            finish = 0
            redlize_points = []
            marginal_points_candidate = []

            for point in marginal_points_sulc:
                sulc = updated_sulc_data[point]

                if sulc >= 0:
                    neighbor_points = [neighbor for neighbor in point_neighbor_points_dict[point] if updated_sulc_data[neighbor] < 0 and curv_data_delete_thicknessZero[neighbor] >= current_curv]

                    for neighbor_point in neighbor_points:
                        second_neighbors = point_neighbor_points_dict[neighbor_point]
                        marginal_gyri_num = sum(1 for second_neighbor in second_neighbors if updated_sulc_data[second_neighbor] < 0 and second_neighbor in marginal_points_gyri)

                        if (marginal_gyri_num < 2) or (marginal_gyri_num == 2 and (np.sum(updated_sulc_data[second_neighbors] < 0) > 2 or set(marginal_gyri_points + [neighbor_point]) in triangle_collection)):
                            redlize_points.append(neighbor_point)
                            marginal_points_candidate = list(set(marginal_points_candidate + second_neighbors))
                            finish = 1

            updated_sulc_data[redlize_points] = 1
            marginal_points, marginal_points_gyri, marginal_points_sulc = find_marginal_point(list(set(marginal_points + marginal_points_candidate)), point_neighbor_points_dict, updated_sulc_data)

            if round_num % 500 == 0:
                write_featured_sphere_from_variable(orig_sphere_polydata, {'sulc_updated': updated_sulc_data}, output_prefix + '_sphere_round(' + str(round_num) + ')_marginal_point.vtk')

    print('绘制最终细GYRI路径在球面上：' + time.asctime(time.localtime(time.time())))
    write_featured_sphere_from_variable(orig_sphere_polydata, {'sulc_updated': updated_sulc_data}, output_prefix + '_sphere_thin.vtk')
    write_featured_sphere_from_variable(orig_surf_polydata, {'sulc_updated': updated_sulc_data}, output_prefix + '_surf_thin.vtk')

    return updated_sulc_data

def find_the_patchSize_of_gyri_point(sulc_data, point_neighbor_points_dict):
    """
    找到每个Gyri点的补丁大小。

    参数:
        sulc_data (numpy.ndarray): Sulc数据数组。
        point_neighbor_points_dict (dict): 顶点之间连通性的字典。

    返回:
        dict: 每个点的补丁大小字典，{点: 补丁大小}。
    """
    point_patchSize_dict = {}

    for point in point_neighbor_points_dict.keys():
        if sulc_data[point] > 0:
            point_patchSize_dict[point] = -1
        else:
            inner_points_list = []
            outer_points_list = [point]
            round_num = 0

            while np.sum(sulc_data[outer_points_list] < 0) == len(outer_points_list):
                round_num += 1
                inner_points_list += outer_points_list
                next_outer_candidate = []

                for outer_point in outer_points_list:
                    neighbor_points = [
                        neighbor_point for neighbor_point in point_neighbor_points_dict[outer_point]
                        if neighbor_point not in inner_points_list
                    ]
                    next_outer_candidate += neighbor_points

                outer_points_list = list(set(next_outer_candidate))

            point_patchSize_dict[point] = round_num

    return point_patchSize_dict


def find_the_max_patchSize_point(unconnected_points_list, point_patchSize_dict_updated):
    """
    找到补丁大小最大的未连接点。

    参数:
        unconnected_points_list (list): 未连接的点列表。
        point_patchSize_dict_updated (dict): 补丁大小字典。

    返回:
        int: 补丁大小最大的点索引。
    """
    max_patchSize_point = max(unconnected_points_list, key=lambda p: point_patchSize_dict_updated[p])
    return max_patchSize_point


def find_the_max_patchSize_candidate_connection(point_patchSize_dict_updated, connected_candidate_dict):
    """
    找到补丁大小最大的连接候选点。

    参数:
        point_patchSize_dict_updated (dict): 补丁大小字典。
        connected_candidate_dict (dict): 候选连接点字典。

    返回:
        list: 补丁大小最大的连接候选点对。
    """
    max_patchSize = -1
    candidate_connection = None
    candidate_connections_list = []

    for point, connections in connected_candidate_dict.items():
        if point_patchSize_dict_updated[point] > max_patchSize:
            max_patchSize = point_patchSize_dict_updated[point]
            candidate_connections_list = connections
        elif point_patchSize_dict_updated[point] == max_patchSize:
            candidate_connections_list += connections

    if len(candidate_connections_list) == 1:
        candidate_connection = candidate_connections_list[0]
    else:
        max_patchSize = -1
        for connection in candidate_connections_list:
            if point_patchSize_dict_updated[connection[1]] > max_patchSize:
                max_patchSize = point_patchSize_dict_updated[connection[1]]
                candidate_connection = connection

    return candidate_connection


def write_skeleton_by_connection_pair(orig_sphere_polydata, connected_lines_list, output):
    """
    根据连接对写入骨架到VTK文件。

    参数:
        orig_sphere_polydata (vtk.vtkPolyData): 原始sphere PolyData对象。
        connected_lines_list (list): 连接对列表。
        output (str): 输出文件路径。
    """
    points_new = vtk.vtkPoints()
    lines_cell_new = vtk.vtkCellArray()
    line_id = 0

    for line in connected_lines_list:
        coordinate1 = orig_sphere_polydata.GetPoints().GetPoint(line[0])
        coordinate2 = orig_sphere_polydata.GetPoints().GetPoint(line[1])
        points_new.InsertNextPoint(coordinate1)
        points_new.InsertNextPoint(coordinate2)

        line = vtk.vtkLine()
        line.GetPointIds().SetNumberOfIds(2)
        line.GetPointIds().SetId(0, line_id * 2)
        line.GetPointIds().SetId(1, line_id * 2 + 1)
        lines_cell_new.InsertNextCell(line)
        line_id += 1

    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points_new)
    polygonPolyData.SetLines(lines_cell_new)
    polygonPolyData.Modified()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polygonPolyData)
    writer.SetFileName(output)
    writer.Write()


def write_all_points_and_skeleton_by_connection_pair(orig_sphere_polydata, connected_lines_list, output):
    """
    根据连接对写入所有点和骨架到VTK文件。

    参数:
        orig_sphere_polydata (vtk.vtkPolyData): 原始sphere PolyData对象。
        connected_lines_list (list): 连接对列表。
        output (str): 输出文件路径。
    """
    point_num = orig_sphere_polydata.GetNumberOfPoints()
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

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polygonPolyData)
    writer.SetFileName(output)
    writer.Write()


def create_gyri_connection_count(point_connect_points_dict_thin_gyri_parts):
    """
    创建Gyri连接计数字典。

    参数:
        point_connect_points_dict_thin_gyri_parts (dict): Gyri部分的顶点连接字典。

    返回:
        dict: Gyri连接计数字典。
    """
    gyri_connection_count = {point: len(neighbors) for point, neighbors in point_connect_points_dict_thin_gyri_parts.items()}
    return gyri_connection_count


def create_tree(orig_sphere_polydata, orig_surf_polydata, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, thin_sulc_data, output_prefix):
    """
    创建树结构，通过连接Gyri部分的点。

    参数:
        orig_sphere_polydata (vtk.vtkPolyData): 原始sphere PolyData对象。
        orig_surf_polydata (vtk.vtkPolyData): 原始surf PolyData对象。
        point_patchSize_dict_updated (dict): 更新后的补丁大小字典。
        point_connect_points_dict_thin_gyri_parts (dict): Gyri部分的顶点连接字典。
        thin_sulc_data (numpy.ndarray): Sulc数据数组。
        output_prefix (str): 输出文件前缀。

    返回:
        tuple: (连接线列表, 父节点字典)
    """
    print('================= 创建树：\t' + time.asctime(time.localtime(time.time())) + '=========================')

    point_num = orig_sphere_polydata.GetNumberOfPoints()
    connected_points_list = []
    connected_lines_list = []
    connected_candidate_dict = defaultdict(list)
    unconnected_points_list = [point for point in range(point_num) if thin_sulc_data[point] < 0]
    father_dict = {}
    gyri_connection_count = create_gyri_connection_count(point_connect_points_dict_thin_gyri_parts)

    while unconnected_points_list:
        print(' 剩余点数：\t', len(unconnected_points_list))

        if not connected_candidate_dict:
            max_patchSize_point = find_the_max_patchSize_point(unconnected_points_list, point_patchSize_dict_updated)
            connected_points_list.append(max_patchSize_point)
            unconnected_points_list.remove(max_patchSize_point)
            father_dict[max_patchSize_point] = max_patchSize_point

        available_connected_points_list = [point for point in connected_points_list if gyri_connection_count[point] > 0]

        for point in available_connected_points_list:
            neighbors = [neighbor for neighbor in point_connect_points_dict_thin_gyri_parts[point] if neighbor not in connected_points_list]
            for neighbor in neighbors:
                connected_candidate_dict[neighbor].append([neighbor, point])

        if connected_candidate_dict:
            candidate = find_the_max_patchSize_candidate_connection(point_patchSize_dict_updated, connected_candidate_dict)
            connected_points_list.append(candidate[0])
            unconnected_points_list.remove(candidate[0])
            connected_lines_list.append(candidate)

            for connection_of_candidate in point_connect_points_dict_thin_gyri_parts[candidate[0]]:
                gyri_connection_count[connection_of_candidate] -= 1

            if candidate[0] not in father_dict:
                father_dict[candidate[0]] = candidate[1]
            else:
                print('Reconnecting the existing point!')
                exit()

    for connection in connected_lines_list:
        if [connection[1], connection[0]] in connected_lines_list:
            print('Connection already in connected_lines_list')
            exit()

    write_skeleton_by_connection_pair(orig_sphere_polydata, connected_lines_list, output_prefix + '_sphere_initial_tree.vtk')
    write_skeleton_by_connection_pair(orig_surf_polydata, connected_lines_list, output_prefix + '_surf_initial_tree.vtk')

    return connected_lines_list, father_dict


def read_connection_of_skeleton_file(skeleton_polydata):
    """
    从骨架PolyData对象中读取连接对。

    参数:
        skeleton_polydata (vtk.vtkPolyData): 骨架PolyData对象。

    返回:
        list: 连接对列表。
    """
    connected_lines_list = []
    line_num = skeleton_polydata.GetNumberOfCells()

    for line in range(line_num):
        point_id1 = skeleton_polydata.GetCell(line).GetPointIds().GetId(0)
        point_id2 = skeleton_polydata.GetCell(line).GetPointIds().GetId(1)
        connected_lines_list.append([point_id1, point_id2])

    return connected_lines_list


def create_tree_connection_dict(connected_lines_list):
    """
    创建树连接字典。

    参数:
        connected_lines_list (list): 连接对列表。

    返回:
        dict: 树连接字典。
    """
    point_lines_dict = defaultdict(set)

    for line in connected_lines_list:
        point_lines_dict[line[0]].add(line[1])
        point_lines_dict[line[1]].add(line[0])

    return point_lines_dict


def step_n_father(point, n, father_dict):
    """
    获取指定点的n级父节点列表。

    参数:
        point (int): 点索引。
        n (int): 父节点级数。
        father_dict (dict): 父节点字典。

    返回:
        list: n级父节点列表。
    """
    point_father_list = []
    current_father = father_dict[point]

    for _ in range(n):
        point_father_list.append(current_father)
        current_father = father_dict[current_father]

    return point_father_list


def get_connection_degree_of_step_n_father(start_point, end_father, father_dict, point_lines_dict):
    """
    获取指定点的n级父节点的连接度。

    参数:
        start_point (int): 起始点索引。
        end_father (int): 终止父节点索引。
        father_dict (dict): 父节点字典。
        point_lines_dict (dict): 点连接字典。

    返回:
        list: n级父节点的连接度列表。
    """
    connection_degrees = []
    father = father_dict[start_point]

    while father != end_father:
        degree = len(point_lines_dict[father])
        connection_degrees.append(degree)
        father = father_dict[father]

    degree = len(point_lines_dict[father])
    connection_degrees.append(degree)

    return connection_degrees


def connect_break_in_circle(point_lines_dict, point_connect_points_dict_thin_gyri_parts, deleted_round1_points, father_dict):
    """
    连接圆圈中的断开点。

    参数:
        point_lines_dict (dict): 点连接字典。
        point_connect_points_dict_thin_gyri_parts (dict): Gyri部分的顶点连接字典。
        deleted_round1_points (list): 删除的第一轮点列表。
        father_dict (dict): 父节点字典。

    返回:
        dict: 更新后的点连接字典。
    """
    for point in point_lines_dict:
        if len(point_lines_dict[point]) == 1:
            success_connection = False
            for neighbor in point_connect_points_dict_thin_gyri_parts[point]:
                if len(point_lines_dict.get(neighbor, [])) == 1:
                    if not set(step_n_father(point, 10, father_dict)).intersection(set(step_n_father(neighbor, 10, father_dict))):
                        point_lines_dict[point].add(neighbor)
                        point_lines_dict[neighbor].add(point)
                        success_connection = True
                        break

            if not success_connection:
                first_round_neighbors = [neighbor for neighbor in point_connect_points_dict_thin_gyri_parts[point] if neighbor in deleted_round1_points]
                for first_deleted_round1_neighbor in first_round_neighbors:
                    for neighbor in point_connect_points_dict_thin_gyri_parts[first_deleted_round1_neighbor]:
                        if len(point_lines_dict.get(neighbor, [])) == 1 and neighbor != point:
                            if not set(step_n_father(point, 10, father_dict)).intersection(set(step_n_father(neighbor, 10, father_dict))):
                                point_lines_dict[point].add(first_deleted_round1_neighbor)
                                point_lines_dict[first_deleted_round1_neighbor] = {point, neighbor}
                                point_lines_dict[neighbor].add(first_deleted_round1_neighbor)
                                success_connection = True
                                break
                            else:
                                print(f'Case-deleted_round1: connect two points belonging to the same branch: {point}, {first_deleted_round1_neighbor}, {neighbor}')
                    if success_connection:
                        break

            if not success_connection:
                print('Remaining endpoint:', point)

    return point_lines_dict


def clear_empty_point_in_dict(point_lines_dict):
    """
    清除字典中无连接的空节点。

    参数:
        point_lines_dict (dict): 点连接字典。

    返回:
        dict: 清除后的点连接字典。
    """
    return {point: neighbors for point, neighbors in point_lines_dict.items() if neighbors}


def trim_all_round1_long(point_lines_dict, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, father_dict, inner_gyri_points):
    """
    删除所有第一轮短分支。

    参数:
        point_lines_dict (dict): 点连接字典。
        point_patchSize_dict_updated (dict): 更新后的补丁大小字典。
        point_connect_points_dict_thin_gyri_parts (dict): Gyri部分的顶点连接字典。
        father_dict (dict): 父节点字典。
        inner_gyri_points (list): 内部Gyri点列表。

    返回:
        tuple: (去掉短分支后的点连接字典, 删除的第一轮点列表)

def find_skeleton(orig_sphere_polydata, orig_surf_polydata, point_patchSize_dict_updated, curv_data_delete_thicknessZero, original_sulc_data, thin_sulc_data, point_neighbor_points_dict, point_connect_points_dict_thin_gyri_parts, connected_lines_list, father_dict, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix):
    """
    构建骨架，通过生成和修剪树结构。

    参数:
        orig_sphere_polydata (vtk.vtkPolyData): 原始sphere PolyData对象。
        orig_surf_polydata (vtk.vtkPolyData): 原始surf PolyData对象。
        point_patchSize_dict_updated (dict): 更新后的补丁大小字典。
        curv_data_delete_thicknessZero (numpy.ndarray): 删除厚度0的曲率数据数组。
        original_sulc_data (numpy.ndarray): 原始Sulc数据数组。
        thin_sulc_data (numpy.ndarray): 更新后的Sulc数据数组。
        point_neighbor_points_dict (dict): 顶点之间连通性的字典。
        point_connect_points_dict_thin_gyri_parts (dict): Gyri部分的顶点连接字典。
        connected_lines_list (list): 已连接的线段列表。
        father_dict (dict): 父节点字典。
        length_thres_of_long_gyri (int): 长Gyri的长度阈值。
        neighbor_missing_path_smallest_step (int): 缺失路径的最小步长。
        flat_threshold_for_convex_gyri (int): 凸形Gyri的平直阈值。
        nearest_skeleton_num (int): 最近骨架点数。
        island_gyri_length_thres (int): 小岛Gyri的下界长度。
        output_prefix (str): 输出文件前缀。

    返回:
        None
    """
    print('================= 构建骨架：\t' + time.asctime(time.localtime(time.time())) + '=======================')

    marginal_points, marginal_points_gyri, marginal_points_sulc = find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, thin_sulc_data)

    print('开始修剪\t' + time.asctime(time.localtime(time.time())))
    tree_point_lines_dict = create_tree_connection_dict(connected_lines_list)

    inner_gyri_points = [gyri_point for gyri_point in tree_point_lines_dict if gyri_point not in marginal_points_gyri]

    initial_skeleton_point_lines_dict, deleted_round1_points = trim_all_round1_long(tree_point_lines_dict, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, father_dict, inner_gyri_points)

    initial_skeleton_connection_list = create_connecting_pair(initial_skeleton_point_lines_dict)

    print('绘制初始骨架\t' + time.asctime(time.localtime(time.time())))
    write_skeleton_by_connection_pair(orig_sphere_polydata, initial_skeleton_connection_list, output_prefix + '_sphere_initial_skeleton.vtk')
    write_skeleton_by_connection_pair(orig_surf_polydata, initial_skeleton_connection_list, output_prefix + '_surf_initial_skeleton.vtk')

    print('连接骨架中的断裂处\t' + time.asctime(time.localtime(time.time())))
    final_point_lines_dict = connect_break_in_circle(initial_skeleton_point_lines_dict, point_connect_points_dict_thin_gyri_parts, deleted_round1_points, father_dict)

    final_connection_list = create_connecting_pair(final_point_lines_dict)
    write_all_points_and_skeleton_by_connection_pair(orig_sphere_polydata, final_connection_list, output_prefix + '_sphere_skeleton_connect_break_allpoints.vtk')
    write_all_points_and_skeleton_by_connection_pair(orig_surf_polydata, final_connection_list, output_prefix + '_surf_skeleton_connect_break_allpoints.vtk')

    final_point_lines_dict = clear_empty_point_in_dict(final_point_lines_dict)

    print('寻找缺失的Gyri\t' + time.asctime(time.localtime(time.time())))
    final_point_lines_dict = create_connection_for_missing_gyri(orig_surf_polydata, orig_sphere_polydata, final_point_lines_dict, original_sulc_data, curv_data_delete_thicknessZero, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix)

    all_connection_list = create_connecting_pair(final_point_lines_dict)
    write_all_points_and_skeleton_by_connection_pair(orig_sphere_polydata, all_connection_list, output_prefix + '_sphere_skeleton_allpoints_final.vtk')
    write_all_points_and_skeleton_by_connection_pair(orig_surf_polydata, all_connection_list, output_prefix + '_surf_skeleton_allpoints_final.vtk')

    all_point_lines_dict = create_tree_connection_dict(all_connection_list)

    print('查找并绘制三齿结构\t' + time.asctime(time.localtime(time.time())))
    hinge3_list = create_3hinge(all_point_lines_dict)
    hinge3_txt = open(output_prefix + '_3hinge_ids.txt', 'w')
    for hinge in hinge3_list:
        hinge3_txt.write(str(hinge) + '\n')
    hinge3_txt.close()
    draw_3hinge_on_surf(orig_sphere_polydata, hinge3_list, output_prefix + '_sphere_3hinge_vertex.vtk')
    draw_3hinge_on_surf(orig_surf_polydata, hinge3_list, output_prefix + '_surf_3hinge_vertex.vtk')

def find_skeleton_missing(orig_sphere_polydata, orig_surf_polydata, skeleton_polydata, curv_data_delete_thicknessZero, original_sulc_data, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix):
    """
    构建并连接缺失的Gyri部分，完成骨架的构建。

    参数:
        orig_sphere_polydata (vtk.vtkPolyData): 原始sphere PolyData对象。
        orig_surf_polydata (vtk.vtkPolyData): 原始surf PolyData对象。
        skeleton_polydata (vtk.vtkPolyData): 骨架PolyData对象。
        curv_data_delete_thicknessZero (numpy.ndarray): 删除厚度0的曲率数据数组。
        original_sulc_data (numpy.ndarray): 原始Sulc数据数组。
        length_thres_of_long_gyri (int): 长Gyri的长度阈值。
        neighbor_missing_path_smallest_step (int): 缺失路径的最小步长。
        flat_threshold_for_convex_gyri (int): 凸形Gyri的平直阈值。
        nearest_skeleton_num (int): 最近骨架点数。
        island_gyri_length_thres (int): 小岛Gyri的下界长度。
        output_prefix (str): 输出文件前缀。

    返回:
        None
    """
    print('================= 构建骨架，并连接缺失的部分：\t' + time.asctime(time.localtime(time.time())) + '=======================')

    final_connection_list = read_connection_of_skeleton_file(skeleton_polydata)
    final_point_lines_dict = create_tree_connection_dict(final_connection_list)

    print('查找缺失的Gyri部分\t' + time.asctime(time.localtime(time.time())))
    final_point_lines_dict = create_connection_for_missing_gyri(
        orig_surf_polydata, orig_sphere_polydata, final_point_lines_dict, 
        original_sulc_data, curv_data_delete_thicknessZero, length_thres_of_long_gyri, 
        neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, 
        nearest_skeleton_num, island_gyri_length_thres, output_prefix
    )

    all_connection_list = create_connecting_pair(final_point_lines_dict)
    write_all_points_and_skeleton_by_connection_pair(orig_sphere_polydata, all_connection_list, output_prefix + '_sphere_skeleton_allpoints_final.vtk')
    write_all_points_and_skeleton_by_connection_pair(orig_surf_polydata, all_connection_list, output_prefix + '_surf_skeleton_allpoints_final.vtk')

    all_point_lines_dict = create_tree_connection_dict(all_connection_list)

    print('查找并绘制三齿结构\t' + time.asctime(time.localtime(time.time())))
    hinge3_list = create_3hinge(all_point_lines_dict)
    hinge3_txt = open(output_prefix + '_3hinge_ids.txt', "w")
    for hinge in hinge3_list:
        hinge3_txt.write(str(hinge) + "\n")
    hinge3_txt.close()
    draw_3hinge_on_surf(orig_sphere_polydata, hinge3_list, output_prefix + '_sphere_3hinge_vertex.vtk')
    draw_3hinge_on_surf(orig_surf_polydata, hinge3_list, output_prefix + '_surf_3hinge_vertex.vtk')