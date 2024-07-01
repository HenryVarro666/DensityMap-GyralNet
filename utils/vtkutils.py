'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-06-25 20:36:56
FilePath: /DensityMap+GNN/utils/vtk_utils.py
'''
import vtk
from utils.featured_sphere import Writer

class Read:
    """
    A utility class for reading VTK files and extracting information from them.
    """

    @classmethod
    def read_vtk_file(cls, vtk_file):
        """
        Reads a VTK file and returns the polydata.

        Args:
            vtk_file (str): The path to the VTK file.

        Returns:
            vtkPolyData: The polydata read from the VTK file.
        """
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
    
    @classmethod
    def read_connection_of_skelenton_file(cls, skelenten_polydata):
        """
        Reads the connection of a skeleton file and returns a list of connected lines.

        Parameters:
        skelenten_polydata (vtkPolyData): The skeleton polydata object.

        Returns:
        connected_lines_list (list): A list of connected lines, where each line is represented by a pair of point IDs.
        """
        connected_lines_list = list()
        line_num = skelenten_polydata.GetNumberOfCells()
        for line in range(line_num):
            pointID1 = skelenten_polydata.GetCell(line).GetPointIds().GetId(0)
            pointID2 = skelenten_polydata.GetCell(line).GetPointIds().GetId(1)
            connected_lines_list.append([pointID1, pointID2])
        return connected_lines_list
    
class Draw:
    '''
    A class that provides methods for drawing various elements using VTK.

    Methods:
        - draw_patchSize_colorful_file: Draws patch sizes on a sphere and surface polydata.
        - draw_3hinge_on_surf: Draws 3-hinge points on a surface polydata.
    '''

    @classmethod
    def draw_patchSize_colorful_file(cls, sphere_polydata, surf_polydata, point_patchSize_dict, output_prefix, type):
        '''
        Draws patch sizes on a sphere and surface polydata.

        Parameters:
            sphere_polydata (vtkPolyData): The sphere polydata.
            surf_polydata (vtkPolyData): The surface polydata.
            point_patchSize_dict (dict): A dictionary mapping point indices to patch sizes.
            output_prefix (str): The output file prefix.
            type (str): The type of data.

        Returns:
            None
        '''
        patchSize_data = list()
        for point in range(sphere_polydata.GetNumberOfPoints()):
            patchSize_data.append(point_patchSize_dict[point])
        Writer.write_featured_sphere_from_variable_single(sphere_polydata, type, patchSize_data, output_prefix + '_sphere_patchSize_' + type + '.vtk')
        Writer.write_featured_sphere_from_variable_single(surf_polydata, type, patchSize_data, output_prefix + '_surf_patchSize_' + type + '.vtk')
    
    @classmethod
    def draw_3hinge_on_surf(cls, surf_polydata, hinge3_list, output_3hinge_vertex):
        '''
        Draws 3-hinge points on a surface polydata.

        Parameters:
            surf_polydata (vtkPolyData): The surface polydata.
            hinge3_list (list): A list of 3-hinge point indices.
            output_3hinge_vertex (str): The output file name.

        Returns:
            None
        '''
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

class ThreeHG:
    @classmethod
    def create_3hinge(cls, final_point_lines_dict):
        """
        Creates a list of 3-hinge points from a dictionary of points and lines.

        Args:
            final_point_lines_dict (dict): A dictionary where the keys are points and the values are lines.

        Returns:
            list: A list of 3-hinge points.
        """
        hinge3_list = list()
        for point in final_point_lines_dict.keys():
            if len(final_point_lines_dict[point]) >= 3:
                hinge3_list.append(point)
        return hinge3_list
    
    @classmethod
    def draw_3hinge_on_surf(cls, surf_polydata, hinge3_list, output_3hinge_vertex):
        """
        Draws the 3-hinge points on a surface and saves the result to a file.

        Args:
            surf_polydata (vtkPolyData): The surface polydata.
            hinge3_list (list): A list of 3-hinge points.
            output_3hinge_vertex (str): The output file path for the 3-hinge points.

        Returns:
            None
        """
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
