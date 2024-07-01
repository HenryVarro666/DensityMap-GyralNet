'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-06-26 17:06:00
FilePath: /DensityMap+GNN/gyrianalyzer/write_featured_sphere_from_file.py
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

class Writer: 
    def __init__(cls):
        pass
    
    @classmethod
    def write_featured_sphere(cls, orig_sphere_polydata, feature_file_dict, output):
        """
        Write the featured sphere to a VTK file.

        Args:
            orig_sphere_polydata: The original sphere polydata.
            feature_file_dict: A dictionary containing feature files for different features.
            output: The output file path.

        Returns:
            None
        """
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
            # feature_name = 'sulc', 'curv', 'thickness'
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

    @classmethod
    def write_featured_sphere_from_variable(cls, orig_sphere_polydata, feature_name_variable_dict, output):
        """
        Write a featured sphere to a VTK file from the given input.

        Parameters:
        - orig_sphere_polydata: The original sphere polydata.
        - feature_name_variable_dict: A dictionary mapping feature names to their corresponding variables.
        - output: The output file path.

        Returns:
        None
        """

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

    @classmethod
    def write_featured_sphere_from_variable_single(cls, orig_sphere_polydata, feature_name, feature_variable, output):
        """
        Write a featured sphere to a VTK file.

        Args:
            orig_sphere_polydata (vtkPolyData): The original sphere polydata.
            feature_name (str): The name of the feature.
            feature_variable (list): The feature variable values.
            output (str): The output file path.

        Returns:
            None
        """
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

    @classmethod
    def find_marginal_point(cls, points_list, point_neighbor_points_dict, sulc_data):
        """
        Finds the marginal points in a list of points based on their neighboring points and sulc data.

        Args:
            points_list (list): A list of points.
            point_neighbor_points_dict (dict): A dictionary mapping each point to its neighboring points.
            sulc_data (numpy.ndarray): An array containing sulc data for each point.

        Returns:
            tuple: A tuple containing three lists - marginal_points, marginal_points_gyri, and marginal_points_sulc.
                - marginal_points: A list of marginal points.
                - marginal_points_gyri: A list of marginal points with negative sulc values.
                - marginal_points_sulc: A list of marginal points with positive sulc values.
        """
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
    
    @classmethod
    def write_thin_gyri_on_sphere_point_marginal_sulc_curv(cls, orig_sphere_polydata, orig_surf_polydata, point_neighbor_points_dict, updated_sulc_data, curv_data_delete_thicknessZero, expend_curv_step_size, output_prefix):
        """
        Writes thin gyri on a sphere based on marginal sulcal curvature.

        Args:
            orig_sphere_polydata: The original sphere polydata.
            orig_surf_polydata: The original surface polydata.
            point_neighbor_points_dict: A dictionary mapping each point to its neighboring points.
            updated_sulc_data: The updated sulcal data.
            curv_data_delete_thicknessZero: The curvature data after deleting thickness zero.
            expend_curv_step_size: The step size for expanding the curvature.
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
        marginal_points, marginal_points_gyri, marginal_points_sulc = cls.find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, updated_sulc_data)
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
    
    @classmethod
    def write_skelenton_by_connectionPair(cls, orig_sphere_polydata, connected_lines_list, output):
        """
        Writes a skeleton by connection pair to a file.

        Parameters:
        - orig_sphere_polydata: The original sphere polydata.
        - connected_lines_list: A list of connected lines.
        - output: The output file name.

        Returns:
        None
        """
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

    @classmethod
    def write_allPoints_and_skelenton_by_connectionPair(cls, orig_sphere_polydata, connected_lines_list, output):
        """
        Writes the points and skeleton of a sphere to a file, based on the given connection pairs.

        Parameters:
        - orig_sphere_polydata: The original sphere polydata.
        - connected_lines_list: A list of connection pairs representing the skeleton of the sphere.
        - output: The output file path to write the polydata.

        Returns:
        None
        """
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
