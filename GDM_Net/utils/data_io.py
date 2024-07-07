'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-07-04 19:18:58
FilePath: /DensityMap-GyralNet/GDM_Net/utils/fileprocessing.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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
import shutil

def create_morph_data():
    """
    Creates morphological data for each subject in the given data directory.

    Returns:
        None
    """
    data_dir = "./100206/100206_recon/surf"
    file_list = os.listdir(data_dir)

    for file in file_list:
        # if '.withGrad.32k_fs_LR.Inner.vtk' not in file:
        if '.withGrad.32k_fs_LR.Inner.vtk' not in file:
            continue
        if 'lh' in file:
            hemi = 'lh'
        elif 'rh' in file:
            hemi = 'rh'
        file_path = os.path.join(data_dir, file)
        surf = pyvista.read(file_path)
        Gradient_Density = surf['gradient_density']
        io.write_morph_data(os.path.join(data_dir, "%s.grad"%(hemi)), Gradient_Density, fnum=327680)

def read_vtk_file(vtk_file):
    """
    Read a VTK file and return the polydata.

    Parameters:
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

def clear_output_folder(output_folder):
    if os.path.exists(output_folder):
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.makedirs(output_folder)

    return print("Clear and create output folder: %s"%(output_folder))

def check_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("Created output folder: %s" % (output_folder))
    else:
        print("Output folder already exists: %s" % (output_folder))

if __name__ == "__main__":
    pass