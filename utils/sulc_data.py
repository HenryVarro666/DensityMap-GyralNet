'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-06-28 14:03:27
FilePath: /DensityMap+GNN/utils/sulc_data.py
'''
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

from utils.neighbor import Neighbor
from utils.points import Points
from utils.featured_sphere import Writer

class SulcData:

    @classmethod
    def thickness_denoise(cls, point_num, point_neighbor_points_dict, thickness_data_raw):
        """
        Denoises the thickness data by filling in zero thickness values with neighboring thickness values.

        Args:
            point_num (int): The total number of points.
            point_neighbor_points_dict (dict): A dictionary mapping each point to its neighboring points.
            thickness_data_raw (numpy.ndarray): The raw thickness data.

        Returns:
            numpy.ndarray: The denoised thickness data.
        """
        thickness_data = thickness_data_raw
        for point in range(point_num):
            if thickness_data[point] == 0:
                step_n = 1
                noise = 0
                while not noise and step_n <= 5:
                    neighbor_list, current_outer_points_list = Neighbor.find_round_n_neighbor(point, step_n, point_neighbor_points_dict)
                    # replace thickness 0 with neighbor thickness
                    if np.sum(thickness_data[current_outer_points_list] > 0) == len(current_outer_points_list):
                        thickness_data[neighbor_list] = 1
                        noise = 1
                    else:
                        step_n += 1
        return thickness_data

    @classmethod
    def find_missing_gyri_by_sulc_and_curv(cls, sulc_data, curv_data, outer_gyri_curv_thres):
        """
        Finds missing gyri based on sulc and curv data.

        Args:
            sulc_data (ndarray): Array containing sulc data.
            curv_data (ndarray): Array containing curv data.
            outer_gyri_curv_thres (float): Threshold value for curv data.

        Returns:
            list: List of indices representing missing gyri points.
        """
        missing_gyri_point_list = list()
        for point in range(sulc_data.shape[0]):
            if sulc_data[point] >= 0 and curv_data[point] < outer_gyri_curv_thres:
                missing_gyri_point_list.append(point)
        return missing_gyri_point_list

    @classmethod
    def find_inner_sulci_by_sulc_and_curv(cls, sulc_data, curv_data, point_neighbor_points_dict, inner_sulci_curv_thres, inner_sulci_round_thres, inner_sulci_neighbor_curv_thres):
        """
        Finds inner sulci points based on sulc and curv data.

        Args:
            sulc_data (ndarray): Array containing sulc data.
            curv_data (ndarray): Array containing curv data.
            point_neighbor_points_dict (dict): Dictionary containing neighbor points for each point.
            inner_sulci_curv_thres (float): Threshold value for curv data.
            inner_sulci_round_thres (float): Threshold value for round neighbor points.
            inner_sulci_neighbor_curv_thres (float): Threshold value for neighbor curv data.

        Returns:
            list: List of inner sulci points.

        """
        inner_sulci_point_list = list()
        for point in range(sulc_data.shape[0]):
            if sulc_data[point] < 0 and curv_data[point] > inner_sulci_curv_thres:
                neighbor_collection, round_n_neighbors = Neighbor.find_round_n_neighbor(point, 
                                                                                        inner_sulci_round_thres, 
                                                                                        point_neighbor_points_dict)
                if np.sum(curv_data[round_n_neighbors] > inner_sulci_neighbor_curv_thres) > 0:
                    for neighbor in neighbor_collection:
                        if curv_data[neighbor] > 0:
                            inner_sulci_point_list.append(neighbor)
        return inner_sulci_point_list

    @classmethod
    def update_sulc_data(cls, original_sulc_data, missing_gyri_point_list, inner_sulci_point_list):
        """
        Update the sulcal data based on the provided lists of missing gyri points and inner sulci points.

        Args:
            original_sulc_data (numpy.ndarray): The original sulcal data.
            missing_gyri_point_list (list): A list of points representing missing gyri.
            inner_sulci_point_list (list): A list of points representing inner sulci.

        Returns:
            numpy.ndarray: The updated sulcal data.

        """

        # Create a new array to store the updated sulcal data
        updated_sulc_data = np.zeros(shape=original_sulc_data.shape)
        for point in range(original_sulc_data.shape[0]):
            if point in missing_gyri_point_list:
                updated_sulc_data[point] = -1
            elif point in inner_sulci_point_list:
                updated_sulc_data[point] = 1
            else:
                updated_sulc_data[point] = original_sulc_data[point]
        return updated_sulc_data

    @classmethod
    def initialize_sulc_data(cls, orig_sphere_polydata, orig_surf_polydata, feature_file_dict, point_neighbor_points_dict, inner_sulci_curv_thres, inner_sulci_round_thres, outer_gyri_curv_thres, inner_sulci_neighbor_curv_thres, output_prefix):
        """
        Initializes the sulc data by updating the sulc values based on curvature and thickness.

        Args:
            orig_sphere_polydata: The original sphere polydata.
            orig_surf_polydata: The original surface polydata.
            feature_file_dict: A dictionary containing the paths to the feature files (thickness, sulc, curv).
            point_neighbor_points_dict: A dictionary containing the neighbor points for each point.
            inner_sulci_curv_thres: The threshold for inner sulci curvature.
            inner_sulci_round_thres: The threshold for inner sulci roundness.
            outer_gyri_curv_thres: The threshold for outer gyri curvature.
            inner_sulci_neighbor_curv_thres: The threshold for inner sulci neighbor curvature.
            output_prefix: The prefix for the output files.

        Returns:
            A tuple containing the updated sulc data, the original sulc data, and the curv data after deleting thickness zero values.
        """
        point_num = orig_sphere_polydata.GetNumberOfPoints()
        cell_num = orig_sphere_polydata.GetNumberOfCells()
        CellArray = orig_sphere_polydata.GetPolys()
        Polygons = CellArray.GetData()

        print('update the sulc value by curv:\t' + time.asctime(time.localtime(time.time())))
        ##############################
        thickness_data_raw = io.read_morph_data(feature_file_dict['thickness'])
        # 厚度数据去噪（将厚度为0的点的厚度值替换为邻居点的厚度值）
        thickness_data = cls.thickness_denoise(point_num, 
                                               point_neighbor_points_dict, 
                                               thickness_data_raw)
        ##############################
        sulc_data_raw = io.read_morph_data(feature_file_dict['sulc'])
        # 删除厚度为0的点的sulc数据（厚度为0的点的sulc值为0）
        sulc_data_delete_thicknessZero = np.where(thickness_data > 0, sulc_data_raw, 0)
        # 将sulc数据转换为二分类形式(大于等于0的点的sulc值为1，小于0的点的sulc值为-1)
        sulc_data_binary = np.where(sulc_data_delete_thicknessZero >= 0, 1, -1)
        # 删除孤立的点(孤立的点的sulc值为0)
        sulc_data = Points.delete_isolated_point(point_num, 
                                                 point_neighbor_points_dict, 
                                                 sulc_data_binary)
        ##############################
        curv_data_raw = io.read_morph_data(feature_file_dict['curv'])
        # 删除厚度为0的点的curv数据（厚度为0的点的curv值为0）
        curv_data_delete_thicknessZero = np.where(thickness_data > 0, curv_data_raw, 0)
        ##############################
        # 根据sulc和curv数据找到缺失的脑回点(脑回的sulc值大于等于0，curv值小于outer_gyri_curv_thres)
        missing_gyri_point_list = cls.find_missing_gyri_by_sulc_and_curv(sulc_data, 
                                                                         curv_data_delete_thicknessZero, 
                                                                         outer_gyri_curv_thres)
        # 根据sulc和curv数据找到内部沟回点(内部沟回的sulc值小于0，curv值大于inner_sulci_curv_thres)
        inner_sulci_point_list = cls.find_inner_sulci_by_sulc_and_curv(sulc_data, 
                                                                       curv_data_delete_thicknessZero, 
                                                                       point_neighbor_points_dict, 
                                                                       inner_sulci_curv_thres, 
                                                                       inner_sulci_round_thres, 
                                                                       inner_sulci_neighbor_curv_thres)
        # 更新sulc数据（缺失的脑回点的sulc值为-1，内部沟回点的sulc值为1）
        updated_sulc_data = cls.update_sulc_data(sulc_data, 
                                                 missing_gyri_point_list, 
                                                 inner_sulci_point_list)

        updated_sulc_data = Points.delete_isolated_point(point_num, point_neighbor_points_dict, updated_sulc_data)

        # 保存更新后的sulc数据
        print('draw updated sulc colorful sphere:\t' + time.asctime(time.localtime(time.time())))
        feature_name_variable_dict = {'thickness_data_raw': thickness_data_raw, 
                                    'thickness_de_noise': thickness_data,
                                    'sulc_data_raw': sulc_data_raw,
                                    'sulc_data_delete_thicknessZero': sulc_data_delete_thicknessZero,
                                    'sulc_data_binary': sulc_data_binary, 
                                    'sulc_data_de_isolation': sulc_data,
                                    'curv_data_raw': curv_data_raw,
                                    'curv_data_delete_thicknessZero': curv_data_delete_thicknessZero,
                                    'updated_sulc_data': updated_sulc_data}
        Writer.write_featured_sphere_from_variable(orig_sphere_polydata, feature_name_variable_dict,
                                            output_prefix + '_sphere_feature_updated.vtk')
        Writer.write_featured_sphere_from_variable(orig_surf_polydata, feature_name_variable_dict,
                                            output_prefix + '_surf_feature_updated.vtk')
        return updated_sulc_data, sulc_data, curv_data_delete_thicknessZero