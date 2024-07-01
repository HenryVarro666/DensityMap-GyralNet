'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-06-28 10:41:57
FilePath: /DensityMap+GNN/utils/neighbor.py
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

class Neighbor:
    
    @classmethod
    def find_round_n_neighbor(cls, point, n, point_neighbor_points_dict):
        """
        Finds the round n neighbors of a given point.

        Args:
            point: The point for which neighbors need to be found.
            n: The number of rounds of neighbors to find.
            point_neighbor_points_dict: A dictionary mapping each point to its neighbor points.

        Returns:
            A tuple containing two lists:
            - neighbor_list: A list of all the neighbors found in the n rounds.
            - current_outer_points_list: A list of the neighbors found in the last round.
        """
        neighbor_list = list()
        current_outer_points_list = list()
        current_outer_points_list.append(point)
        while n > 0:
            neighbor_list = list(set(neighbor_list + current_outer_points_list))
            next_outer_points_list = list()
            for point in current_outer_points_list:
                neighbors = [neighbor for neighbor in point_neighbor_points_dict[point] if neighbor not in neighbor_list]
                next_outer_points_list = next_outer_points_list + neighbors
            next_outer_points_list = list(set(next_outer_points_list))
            current_outer_points_list = next_outer_points_list
            n = n - 1
        neighbor_list = list(set(neighbor_list + current_outer_points_list))
        return neighbor_list, current_outer_points_list
        ## 1. neighbor_list: A list of all the neighbors found in the n rounds.
        ## 2. current_outer_points_list: A list of the neighbors found in the last round.
