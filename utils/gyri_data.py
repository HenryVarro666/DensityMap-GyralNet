'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-06-25 20:07:39
FilePath: /DensityMap+GNN/utils/gyral_data.py
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

from utils.featured_sphere import Writer
from utils.points import Points

class GyralData:
    @classmethod
    def find_the_patchSize_of_gyri_point(cls, sulc_data, point_neighbor_points_dict):
        """
        Finds the patch size of each gyri point based on the sulc_data and point_neighbor_points_dict.

        Parameters:
        sulc_data (numpy.ndarray): An array containing sulc data for each point.
        point_neighbor_points_dict (dict): A dictionary containing the neighbor points for each point.

        Returns:
        dict: A dictionary containing the patch size for each gyri point.
        """
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

    @classmethod
    def write_thin_gyri_on_sphere_point_marginal_sulc_curv(cls, orig_sphere_polydata, orig_surf_polydata, point_neighbor_points_dict, updated_sulc_data, curv_data_delete_thicknessZero, expend_curv_step_size, output_prefix):
        """
        Writes thin gyri on a sphere based on marginal sulcal curvature.

        Args:
            orig_sphere_polydata: The original sphere polydata.
            orig_surf_polydata: The original surface polydata.
            point_neighbor_points_dict: A dictionary mapping each point to its neighboring points.
            updated_sulc_data: The updated sulcal data.
            curv_data_delete_thicknessZero: The curvature data with thickness zero deleted.
            expend_curv_step_size: The step size for expanding the gyri.
            output_prefix: The prefix for the output file names.

        Returns:
            The updated sulcal data.
        """
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
        marginal_points, marginal_points_gyri, marginal_points_sulc = Points.find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, updated_sulc_data)
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
                marginal_points, marginal_points_gyri, marginal_points_sulc = Points.find_marginal_point(list(set(marginal_points + marginal_points_candidate)), point_neighbor_points_dict, updated_sulc_data)
                # pdb.set_trace()
                if round % 500 == 0:
                    Writer.write_featured_sphere_from_variable(orig_sphere_polydata, {'sulc_updated': updated_sulc_data}, output_prefix + '_sphere_round(' + str(round) + ')_marginal_point.vtk')

        print('draw final thin path on sphere:\t' + time.asctime(time.localtime(time.time())))
        Writer.write_featured_sphere_from_variable(orig_sphere_polydata, {'sulc_updated': updated_sulc_data}, output_prefix + '_sphere_thin.vtk')
        Writer.write_featured_sphere_from_variable(orig_surf_polydata, {'sulc_updated': updated_sulc_data}, output_prefix + '_surf_thin.vtk')
        return updated_sulc_data
    

    