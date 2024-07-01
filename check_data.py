'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-06-28 12:34:28
FilePath: /DensityMap+GNN/check_data.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import vtk
import nibabel.freesurfer.io as io
from collections import defaultdict
import pdb
import numpy as np
import os
import shutil
import time
import argparse

from utils.points import Points
from utils.vtkutils import Draw, Read

sphere_file = '/mnt/d/DensityMap+GNN/100206/100206_recon/surf/lh.withGrad.32k_fs_LR.Sphere.vtk'
surf_file = '/mnt/d/DensityMap+GNN/100206/100206_recon/surf/lh.withGrad.32k_fs_LR.Inner.vtk'

feature_file_dict = {'sulc': '/mnt/d/DensityMap+GNN/100206/100206_recon/surf/lh.flip.sulc', 'curv': '/mnt/d/DensityMap+GNN/100206/100206_recon/surf/lh.flip.curv', 'thickness': '/mnt/d/DensityMap+GNN/100206/100206_recon/surf/lh.thickness'}

sphere_polydata = Read.read_vtk_file(sphere_file)
surf_polydata = Read.read_vtk_file(surf_file)

sulc_file = '/mnt/d/DensityMap+GNN/100206/100206_recon/surf/lh.flip.sulc'

# def show_featured_sphere(cls, orig_sphere_polydata, feature_file_dict):
#         point_num = orig_sphere_polydata.GetNumberOfPoints()
#         cell_num = orig_sphere_polydata.GetNumberOfCells()

#         # print(point_num)
#         # print(cell_num)

#         for point in range(point_num):
#             coordinate = orig_sphere_polydata.GetPoints().GetPoint(point)
#             # print(str(coordinate[0]) + " " + str(coordinate[1]) + " " + str(coordinate[2]) + '\n')

#         CellArray = orig_sphere_polydata.GetPolys()
#         Polygons = CellArray.GetData()
#         for i in range(0, cell_num):
#             triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
#             print(str(3) + " " + str(triangle[0]) + " " + str(triangle[1]) + " " + str(triangle[2]) + '\n')

#         # scale_dict = {'sulc': 0.0, 'curv': 0.0, 'thickness': 0.00000000001}
#         # for feature_name in scale_dict.keys():
#         #     # print(feature_name)
#         #     # feature_name = 'sulc', 'curv', 'thickness'
#         #     feature_file = feature_file_dict[feature_name]
#         #     features = io.read_morph_data(feature_file)

#         #     for point in range(point_num):
#         #         print(str(features[point]) + '\n')

# def get_connect_points(surf_polydata):
#     point_connect_points_dict = defaultdict(list)
#     cell_num = surf_polydata.GetNumberOfCells()
#     CellArray = surf_polydata.GetPolys()
#     Polygons = CellArray.GetData()

#     # for i in range(cell_num):
#     for i in [0,1,2,3,4,5,6]:
#         triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
#         # print(triangle)
#         if triangle[1] not in point_connect_points_dict[triangle[0]]:
#             point_connect_points_dict[triangle[0]].append(triangle[1])
#         if triangle[2] not in point_connect_points_dict[triangle[0]]:
#             point_connect_points_dict[triangle[0]].append(triangle[2])
#         if triangle[0] not in point_connect_points_dict[triangle[1]]:
#             point_connect_points_dict[triangle[1]].append(triangle[0])
#         if triangle[2] not in point_connect_points_dict[triangle[1]]:
#             point_connect_points_dict[triangle[1]].append(triangle[2])
#         if triangle[0] not in point_connect_points_dict[triangle[2]]:
#             point_connect_points_dict[triangle[2]].append(triangle[0])
#         if triangle[1] not in point_connect_points_dict[triangle[2]]:
#             point_connect_points_dict[triangle[2]].append(triangle[1])
#         print(point_connect_points_dict)
#     return point_connect_points_dict



def read_sulc(sulc_file):
    sulc = io.read_morph_data(sulc_file)
    return sulc
if __name__ == '__main__':
    #  show_featured_sphere(Points, sphere_polydata, feature_file_dict)

    # get_connect_points(Points, surf_polydata)
    # get_connect_points(surf_polydata)

    sulc = read_sulc(sulc_file)
    print(sulc)
    print(sulc.shape)
    print(sulc.shape[0])
    