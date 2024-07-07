from collections import defaultdict
import pyvista
import nibabel
import hcp_utils
import numpy as np
from tqdm import tqdm
import nibabel.freesurfer.io as io
import os
import time
import vtk

def get_connect_points(surf_polydata):
    """
    Returns a dictionary that maps each point in the surface polydata to a list of its connected points.

    Parameters:
    surf_polydata (vtkPolyData): The surface polydata object.

    Returns:
    dict: A dictionary where the keys are the points in the surface polydata and the values are lists of connected points.
    """
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
# def test():
#     scale_dict = {'gradient_density': 0}
#     print(scale_dict['gradient_density'])

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

    ### Write binary features
    ### If the feature value is larger than the threshold, set it to 1; if it is smaller than the negative threshold, set it to -1; otherwise, set it to 0. 
    f.write("POINT_DATA " + str(point_num) + '\n')
    # scale_dict = {'sulc': 0.0, 'curv': 0.0, 'thickness': 0.00000000001, 'gradient_density': 0.0}
    scale_dict = {'gradient_density': 0}

    for feature_name in scale_dict.keys():
        feature_file = feature_file_dict[feature_name]
        features = io.read_morph_data(feature_file)

        f.write("SCALARS " + feature_name + '_binary' + " float" + '\n')
        f.write("LOOKUP_TABLE " + feature_name + '_binary' + '\n')

        # 存在0的情况
        for point in range(point_num):
            if features[point] >= scale_dict[feature_name]:
                f.write(str(1) + '\n')
            # elif features[point] < -1*scale_dict[feature_name]:
            elif features[point] < scale_dict[feature_name]:
                f.write(str(-1) + '\n')
            else:
                f.write(str(0) + '\n')

        # # 不存在0的情况
        # for point in range(point_num):
        #     if features[point] >= scale_dict[feature_name]:
        #         f.write(str(1) + '\n')
        #     else:
        #         f.write(str(-1) + '\n')

        f.write("SCALARS " + feature_name + " float" + '\n')
        f.write("LOOKUP_TABLE " + feature_name + '\n')

        for point in range(point_num):
            f.write(str(features[point]) + '\n')

    f.close()

def write_featured_sphere_from_variable(orig_polydata, feature_name_variable_dict, output):
    point_num = orig_polydata.GetNumberOfPoints()
    cell_num = orig_polydata.GetNumberOfCells()

    f = open(output, 'w')
    f.write("# vtk DataFile Version 3.0\n")
    f.write("mesh surface\n")
    f.write("ASCII\n")
    f.write("DATASET POLYDATA\n")
    f.write("POINTS " + str(point_num) + " float\n")

    for point in range(point_num):
        coordinate = orig_polydata.GetPoints().GetPoint(point)
        f.write(str(coordinate[0]) + " " + str(coordinate[1]) + " " + str(coordinate[2]) + '\n')

    f.write("POLYGONS " + str(cell_num) + " " + str(4 * cell_num) + '\n')
    CellArray = orig_polydata.GetPolys()
    Polygons = CellArray.GetData()
    for i in range(0, cell_num):
        triangle = [Polygons.GetValue(j) for j in range(i * 4 + 1, i * 4 + 4)]
        f.write(str(3) + " " + str(triangle[0]) + " " + str(triangle[1]) + " " + str(triangle[2]) + '\n')

    # # Single
    # f.write("POINT_DATA " + str(point_num) + '\n')
    # f.write("SCALARS " + feature_name + " float" + '\n')
    # f.write("LOOKUP_TABLE " + feature_name + '\n')

    # for point in range(point_num):
    #     f.write(str(feature_variable[point]) + '\n')
    # f.close()

    # Multiple
    f.write("POINT_DATA " + str(point_num) + '\n')
    for feature_name in feature_name_variable_dict.keys():
        feature_variable = feature_name_variable_dict[feature_name]
        f.write("SCALARS " + feature_name + " float" + '\n')
        f.write("LOOKUP_TABLE " + feature_name + '\n')

        for point in range(point_num):
            f.write(str(feature_variable[point]) + '\n')
    f.close()

def write_featured_sphere_from_variable_single(orig_sphere_polydata, feature_variable, feature_name, output):
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

def draw_patchSize_colorful_file(sphere_polydata, surf_polydata, point_patchSize_dict, output_prefix, sphere, feature_name='updated'):
    patchSize_data = list()
    for point in range(sphere_polydata.GetNumberOfPoints()):
        patchSize_data.append(point_patchSize_dict[point])

    sphere_output = os.path.join(output_prefix, "%s_sphere_patchSize_%s.vtk"%(sphere, feature_name))
    surf_output = os.path.join(output_prefix, "%s_surface_patchSize_%s.vtk"%(sphere, feature_name))
    write_featured_sphere_from_variable_single(sphere_polydata, patchSize_data, feature_name, sphere_output)                                                 
    write_featured_sphere_from_variable_single(surf_polydata, patchSize_data, feature_name, surf_output)

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

if __name__=="__main__":
    print("True")