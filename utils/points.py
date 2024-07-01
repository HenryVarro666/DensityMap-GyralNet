'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-06-28 14:48:19
FilePath: /DensityMap+GNN/utils/get_connect_points.py
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


class Points:
    @classmethod
    def get_connect_points(cls, surf_polydata):
        """
        Returns a dictionary that maps each point to a list of its connected points.

        Parameters:
        - surf_polydata: The surface polydata object.

        Returns:
        - point_connect_points_dict: A dictionary where the keys are points and the values are lists of connected points.
        """
        point_connect_points_dict = defaultdict(list)
        cell_num = surf_polydata.GetNumberOfCells()
        CellArray = surf_polydata.GetPolys()
        Polygons = CellArray.GetData()
        """
        代码的目的是构建一个字典,其中键是三角形的顶点,值是与该顶点相连的其他顶点的列表。代码通过遍历三角形并将顶点两两相连来实现这一目的。

        首先,我们获取三角形的数量和三角形的顶点数据。
        然后,我们使用一个循环来遍历每个三角形。
        对于每个三角形,我们从顶点数据中提取三个顶点,并将它们存储在triangle列表中。
        接下来,我们检查每个顶点与其他顶点的连接关系,并将连接关系添加到point_connect_points_dict字典中。
        如果某个顶点与另一个顶点尚未连接,我们将其添加到对应的连接列表中。
        最后,函数返回构建好的point_connect_points_dict字典,其中包含了每个顶点与其相连的其他顶点的信息。

        这段代码使用了defaultdict的特性,使得我们无需在每次添加新键时手动创建空列表。
        如果某个键不存在,defaultdict会自动使用默认工厂函数创建一个默认值,并将其与键关联起来。
        """
        for i in range(cell_num):
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
        ## TODO: 使用pickle来保存dict里的连接数据

    @classmethod
    def find_marginal_point(cls, points_list, point_neighbor_points_dict, sulc_data):
        """
        Find marginal points based on the given criteria.

        Args:
            points_list (list): List of points to check.
            point_neighbor_points_dict (dict): Dictionary mapping each point to its neighbor points.
            sulc_data (numpy.ndarray): Array containing sulc data for each point.

        Returns:
            tuple: A tuple containing three lists:
                - marginal_points: List of all marginal points.
                - marginal_points_gyri: List of marginal points in gyri.
                - marginal_points_sulc: List of marginal points in sulci.
        """
        marginal_points = list()
        marginal_points_gyri = list()
        marginal_points_sulc = list()
        # For each point in the list, check if it is a marginal point (i.e., has neighbors with different signs)
        # 三分类
        for point in points_list:
            neighbor_points = point_neighbor_points_dict[point]
            neighbor_sulcs = sulc_data[neighbor_points]
            # If not all neighbors have the same sign, the point is a marginal point
            if np.sum(neighbor_sulcs < 0) != len(neighbor_points) and np.sum(neighbor_sulcs > 0) != len(neighbor_points):
                marginal_points.append(point)
                # Check if the point is in a gyri or sulcus based on the sign of its sulc value
                if sulc_data[point] < 0:
                    marginal_points_gyri.append(point)
                else:
                    marginal_points_sulc.append(point)
        # Return the list of marginal points and the list of marginal points in gyri and sulci
        return marginal_points, marginal_points_gyri, marginal_points_sulc
    
    @classmethod
    def delete_isolated_point(cls, point_num, point_neighbor_points_dict, sulc_data):
        """
        Deletes isolated points from the sulc_data array based on certain conditions.

        Args:
            cls (class): The class object.
            point_num (int): The total number of points.
            point_neighbor_points_dict (dict): A dictionary mapping each point to its neighbor points.
            sulc_data (ndarray): An array containing sulc values for each point.

        Returns:
            ndarray: The updated sulc_data array after deleting isolated points.
        """
        marginal_points, marginal_points_gyri, marginal_points_sulc = cls.find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, sulc_data)
        for point in range(point_num):
            sulc = sulc_data[point]
            '''isolated point'''
            # If the point is isolated and has a different sign from its neighbors, update the sulc value
            nerighbor_sulcs = sulc_data[point_neighbor_points_dict[point]]
            if sulc > 0 and np.sum(nerighbor_sulcs < 0) == nerighbor_sulcs.shape[0]:
                sulc_data[point] = -1
            elif sulc < 0 and np.sum(nerighbor_sulcs > 0) == nerighbor_sulcs.shape[0]:
                sulc_data[point] = 1
            else:
                '''isolated one-round patch'''
                # If the point is isolated and has neighbors in the marginal points list, update the sulc value
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
    
    @classmethod
    def get_connect_points_gyri_part(cls, surf_polydata, sulc_data):
        """
        Returns a dictionary that maps each point to a list of connected points in the gyri part of the surface.

        Parameters:
        - surf_polydata: The surface polydata.
        - sulc_data: The sulc data.

        Returns:
        - point_connect_points_dict_gyri_parts: A dictionary mapping each point to a list of connected points in the gyri part.
        """
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
    
    @classmethod
    def find_the_max_patchSize_candidate_connection(cls, point_patchSize_dict_updated, connected_candidate_dict):
        """
        Finds the candidate connection with the maximum patch size.

        Args:
            point_patchSize_dict_updated (dict): A dictionary mapping points to their corresponding patch sizes.
            connected_candidate_dict (dict): A dictionary mapping points to a list of candidate connections.

        Returns:
            tuple: The candidate connection with the maximum patch size.

        """
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

