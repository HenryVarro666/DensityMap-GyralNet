'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-07-06 12:51:32
FilePath: /DensityMap-GyralNet/GDM_Net/utils/augmentation.py
'''
import nibabel.freesurfer.io as io
import numpy as np
import pyvista
import vtk
import os
from collections import defaultdict
import time
import shutil

def find_marginal_point(points_list, point_neighbor_points_dict, grad_data_updated):
    marginal_points = list()
    marginal_points_gyri = list()
    marginal_points_sulc = list()
    for point in points_list:
        neighbor_points = point_neighbor_points_dict[point]
        neighbor_grads = grad_data_updated[neighbor_points]
        if np.sum(neighbor_grads < 0) != len(neighbor_points) and np.sum(neighbor_grads > 0) != len(neighbor_points):
            marginal_points.append(point)
            if grad_data_updated[point] < 0:
                marginal_points_gyri.append(point)
            else:
                # grad_data_updated[point] > 0
                marginal_points_sulc.append(point)
                
    return marginal_points, marginal_points_gyri, marginal_points_sulc

def delete_isolated_point(point_num, point_neighbor_points_dict, grad_value_binary):
    marginal_points, marginal_points_gyri, marginal_points_sulc = find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, grad_value_binary)
    for point in range(point_num):
        grad = grad_value_binary[point]
        neighbor_grad = grad_value_binary[point_neighbor_points_dict[point]]
        
        # isolated point
        if grad > 0 and np.sum(neighbor_grad < 0) == neighbor_grad.shape[0]:
            grad_value_binary[point] = -1
        elif grad < 0 and np.sum(neighbor_grad > 0) == neighbor_grad.shape[0]:
            grad_value_binary[point] = 1
        else:
            '''isolated one-round patch'''
            if grad > 0:
                first_neighbor_list = [first_neighbor for first_neighbor in point_neighbor_points_dict[point] if first_neighbor not in marginal_points_gyri]
            elif grad < 0:
                first_neighbor_list = [first_neighbor for first_neighbor in point_neighbor_points_dict[point] if first_neighbor not in marginal_points_sulc]

            second_neighbor_list = list()
            for first_neighbor in first_neighbor_list:
                second_neighbors = [second_neighbor for second_neighbor in point_neighbor_points_dict[first_neighbor] if second_neighbor not in [point] + first_neighbor_list]
                second_neighbor_list = second_neighbor_list + second_neighbors
            second_neighbor_list = list(set(second_neighbor_list))

            if grad > 0 and np.sum(grad_value_binary[second_neighbor_list] < 0) == len(second_neighbor_list):
                grad_value_binary[[point] + first_neighbor_list] = -1
            elif grad < 0 and np.sum(grad_value_binary[second_neighbor_list] > 0) == len(second_neighbor_list):
                grad_value_binary[[point] + first_neighbor_list] = 1


            # # one round neighbors
            # first_neighbor_list = [first_neighbor for first_neighbor in point_neighbor_points_dict[point] if first_neighbor not in marginal_points]
            # if grad > 0 and np.all(grad_value_binary[first_neighbor_list] < 0):
            #     grad_value_binary[[point] + first_neighbor_list] = -1
            # elif grad < 0 and np.all(grad_value_binary[first_neighbor_list] > 0):
            #     grad_value_binary[[point] + first_neighbor_list] = 1

    return grad_value_binary
