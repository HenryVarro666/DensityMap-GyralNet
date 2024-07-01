'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-06-25 20:28:22
FilePath: /DensityMap+GNN/utils/connection.py
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

from utils.vtkutils import Draw, ThreeHG
from utils.featured_sphere import Writer
from utils.points import Points
from utils.neighbor import Neighbor

class Connection:
    @classmethod
    def read_connection_of_skelenton_file(cls, skelenten_polydata):
        """
        Reads the connection of a skeleton file and returns a list of connected lines.

        Args:
            skelenten_polydata: The skeleton polydata object.

        Returns:
            connected_lines_list: A list of connected lines, where each line is represented by a pair of point IDs.

        """
        connected_lines_list = list()
        line_num = skelenten_polydata.GetNumberOfCells()
        for line in range(line_num):
            pointID1 = skelenten_polydata.GetCell(line).GetPointIds().GetId(0)
            pointID2 = skelenten_polydata.GetCell(line).GetPointIds().GetId(1)
            connected_lines_list.append([pointID1, pointID2])
        return connected_lines_list

    @classmethod
    def create_gyri_connection_count(cls, point_connect_points_dict_thin_gyri_parts):
        """
        Calculates the number of connections for each point in the given dictionary.

        Args:
            point_connect_points_dict_thin_gyri_parts (dict): A dictionary where the keys are points and the values are lists of connected points.

        Returns:
            dict: A dictionary where the keys are points and the values are the number of connections for each point.
        """
        gyri_connection_count = dict()
        for point in point_connect_points_dict_thin_gyri_parts.keys():
            count = len(point_connect_points_dict_thin_gyri_parts[point])
            gyri_connection_count[point] = count
        return gyri_connection_count

    @classmethod
    def create_tree_connection_dict(cls, connected_lines_list):
        """
        Creates a dictionary that represents the connections between points in a tree.

        Args:
            connected_lines_list (list): A list of tuples representing the connected lines.

        Returns:
            dict: A dictionary where the keys are the points and the values are sets of connected points.
        """
        point_lines_dict = defaultdict(set)
        for line in connected_lines_list:
            if line[0] not in point_lines_dict.keys():
                point_lines_dict[line[0]] = {line[1]}
            else:
                point_lines_dict[line[0]].add(line[1])
            if line[1] not in point_lines_dict.keys():
                point_lines_dict[line[1]] = {line[0]}
            else:
                point_lines_dict[line[1]].add(line[0])
        return point_lines_dict
    
    @classmethod
    def get_connection_degree_of_step_n_father(cls, start_point, end_father, father_dict, point_lines_dict):
        """
        Calculate the connection degrees between a start point and an end father in a graph.

        Parameters:
        - start_point (int): The starting point in the graph.
        - end_father (int): The target father node in the graph.
        - father_dict (dict): A dictionary mapping each point to its corresponding father node.
        - point_lines_dict (dict): A dictionary mapping each point to a list of lines connected to it.

        Returns:
        - connection_degrees (list): A list of connection degrees between the start point and the end father.
        """
        connection_degrees = list()
        father = father_dict[start_point]
        while father != end_father:
            degree = len(point_lines_dict[father])
            connection_degrees.append(degree)
            father = father_dict[father]
        degree = len(point_lines_dict[father])
        connection_degrees.append(degree)
        return connection_degrees
    
    @classmethod
    def get_connect_points_gyri_part(cls, surf_polydata, sulc_data):
        """
        Returns a dictionary containing the connected points for each gyri part.

        Parameters:
        - surf_polydata: The surface polydata.
        - sulc_data: The sulc data.

        Returns:
        - point_connect_points_dict_gyri_parts: A dictionary where the keys are the points and the values are lists of connected points.
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
    def get_connect_points(cls, surf_polydata):
        """
        Returns a dictionary that maps each point in the given surface polydata to a list of its connected points.

        Parameters:
        - surf_polydata: The surface polydata object.

        Returns:
        - point_connect_points_dict: A dictionary where the keys are the points in the surface polydata and the values are lists of connected points.
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

    @classmethod
    def find_the_patchSize_of_gyri_point(cls, sulc_data, point_neighbor_points_dict):
        """
        Calculates the patch size for each gyri point based on the sulc_data and point_neighbor_points_dict.

        Args:
            sulc_data (numpy.ndarray): An array containing sulc data for each point.
            point_neighbor_points_dict (dict): A dictionary mapping each point to its neighbor points.

        Returns:
            dict: A dictionary mapping each point to its corresponding patch size.

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
    def select_convex_endpoint_for_each_missing_gyri(cls, long_path_convex_points_list, convex_marginal_points_length_dict, curv_data_delete_thicknessZero, point_connect_points_dict_gyri_parts, neighbor_missing_path_smallest_step):
        """
        Selects the convex endpoint for each missing gyri.

        Args:
            long_path_convex_points_list (list): List of convex points in the long path.
            convex_marginal_points_length_dict (dict): Dictionary containing the length of marginal points for each convex point.
            curv_data_delete_thicknessZero (dict): Dictionary containing curvature data for each point.
            point_connect_points_dict_gyri_parts (dict): Dictionary containing the connected points for each gyri part.
            neighbor_missing_path_smallest_step (int): The smallest step for finding neighboring missing paths.

        Returns:
            list: Updated list of long path convex points after selecting the convex endpoint for each missing gyri.
        """
        print('select convex endpoint for each missing gyri \t' + time.asctime(time.localtime(time.time())))
        delete_list = list()
        for convex_point in long_path_convex_points_list:
            point_connection = [point for point in convex_marginal_points_length_dict[convex_point][1].keys() if point in long_path_convex_points_list]
            current_point = point_connection[0]
            max_path = convex_marginal_points_length_dict[current_point][0]
            step_n_neighbor_list, _ = Neighbor.find_round_n_neighbor(convex_point, neighbor_missing_path_smallest_step, point_connect_points_dict_gyri_parts)
            for point in point_connection:
                if point in step_n_neighbor_list:
                    path = convex_marginal_points_length_dict[point][0]
                    if path < max_path:
                        delete_list.append(point)
                    elif path > max_path:
                        max_path = path
                        delete_list.append(current_point)
                        current_point = point
                    else:
                        if curv_data_delete_thicknessZero[point] < curv_data_delete_thicknessZero[current_point]:
                            delete_list.append(current_point)
                            current_point = point
        delete_list = list(set(delete_list))
        for point in delete_list:
            long_path_convex_points_list.remove(point)
        return long_path_convex_points_list

    @classmethod
    def find_convex_marginal_points_of_long_gyri(cls, orig_sphere_polydata, final_point_lines_dict, point_neighbor_points_dict, point_connect_points_dict_gyri_parts, marginal_points_gyri, original_sulc_data, curv_data_delete_thicknessZero, point_patchSize_dict_orig, length_thres_of_long_gyri, nearest_skeleton_num, island_gyri_length_thres):
        """
        Finds the convex marginal points of long gyri.

        Args:
            orig_sphere_polydata: The original sphere polydata.
            final_point_lines_dict: A dictionary containing the final point lines.
            point_neighbor_points_dict: A dictionary containing the neighbor points for each point.
            point_connect_points_dict_gyri_parts: A dictionary containing the connect points for each point in gyri parts.
            marginal_points_gyri: A list of marginal points in gyri.
            original_sulc_data: The original sulc data.
            curv_data_delete_thicknessZero: The curv data after deleting thickness zero.
            point_patchSize_dict_orig: A dictionary containing the patch size for each point in the original sphere.
            length_thres_of_long_gyri: The length threshold for long gyri.
            nearest_skeleton_num: The number of nearest skeleton points.
            island_gyri_length_thres: The length threshold for island gyri.

        Returns:
            A tuple containing the following:
            - long_convax_marginal_points_list: A list of long convex marginal points.
            - convex_marginal_points_length_dict: A dictionary containing the length information for each convex marginal point.
            - island_convex_marginal_points_length_dict: A dictionary containing the length information for each convex marginal point in island gyri.
        """
        print('find convex marginal points of long gyri  \t' + time.asctime(time.localtime(time.time())))
        convex_marginal_points_gyri = list()
        for point in marginal_points_gyri:
            first_gyri_neighbor = [first_neighbor for first_neighbor in point_neighbor_points_dict[point] if original_sulc_data[first_neighbor] < 0]
            first_sulc_neighbor = [first_neighbor for first_neighbor in point_neighbor_points_dict[point] if original_sulc_data[first_neighbor] > 0]
            if (len(first_gyri_neighbor) - 2 < len(first_sulc_neighbor)) and curv_data_delete_thicknessZero[point] < 0:
                convex_marginal_points_gyri.append(point)
        convex_marginal_points_length_dict = defaultdict(list)
        long_convax_marginal_points_list = list()
        island_convex_marginal_points_length_dict = defaultdict(list)
        for convex_marginal_point in convex_marginal_points_gyri:
            from_marginal_to_skelenton_father_sons_dict = defaultdict(list)
            from_skelenton_to_marginal_son_fathers_dict = defaultdict(list)
            path = 0
            reach_skelenton = 0
            max_step_after_reach_skelenton = 6
            skelenton_point_set = set()
            current_outer_points_list = list()
            current_outer_points_list.append(convex_marginal_point)
            while max_step_after_reach_skelenton and (not reach_skelenton):
                next_outer_points_list = list()
                for point in current_outer_points_list:
                    neighbors = [neighbor for neighbor in point_connect_points_dict_gyri_parts[point] if neighbor not in from_marginal_to_skelenton_father_sons_dict.keys() and neighbor not in current_outer_points_list and curv_data_delete_thicknessZero[neighbor] < 0]
                    if point in from_marginal_to_skelenton_father_sons_dict.keys():
                        print('from_marginal_to_skelenton_father_sons_dict error!')
                    else:
                        from_marginal_to_skelenton_father_sons_dict[point] = neighbors
                    for neighbor in neighbors:
                        if neighbor not in from_skelenton_to_marginal_son_fathers_dict.keys():
                            from_skelenton_to_marginal_son_fathers_dict[neighbor] = [point]
                        else:
                            from_skelenton_to_marginal_son_fathers_dict[neighbor].append(point)

                    next_outer_points_list = next_outer_points_list + neighbors

                if len(next_outer_points_list):

                    path = path + 1

                    # test single point
                    # if convex_marginal_point == 71363:
                    #     draw_3hinge_on_surf(orig_sphere_polydata, next_outer_points_list,
                    #                         output_prefix + '_71363_path_' + str(path) + '.vtk')
                    #     print(path)

                    skelenton_point_set = skelenton_point_set | (set(next_outer_points_list) & set(final_point_lines_dict.keys()))
                    if skelenton_point_set:
                        if max_step_after_reach_skelenton == 6:
                            real_path = path
                            averaged_patchSize = 0
                            for skelenton_point in skelenton_point_set:
                                averaged_patchSize = averaged_patchSize + point_patchSize_dict_orig[skelenton_point]
                            averaged_patchSize = averaged_patchSize / len(skelenton_point_set)
                        max_step_after_reach_skelenton -= 1
                    if len(skelenton_point_set) >= nearest_skeleton_num:
                        reach_skelenton = 1
                        if real_path - averaged_patchSize > length_thres_of_long_gyri:
                            convex_marginal_points_length_dict[convex_marginal_point] = [real_path, from_marginal_to_skelenton_father_sons_dict, from_skelenton_to_marginal_son_fathers_dict]
                            long_convax_marginal_points_list.append(convex_marginal_point)
                    else:
                        current_outer_points_list = set(next_outer_points_list)
                else:
                    if path > island_gyri_length_thres:
                        island_convex_marginal_points_length_dict[convex_marginal_point] = [path, from_marginal_to_skelenton_father_sons_dict, from_skelenton_to_marginal_son_fathers_dict]
                    break

        return long_convax_marginal_points_list, convex_marginal_points_length_dict, island_convex_marginal_points_length_dict

    @classmethod
    def find_connections_for_long_island_convex_marginal_point(cls, long_convex_marginal_point, point_connect_points_dict_gyri_parts, curv_data_delete_thicknessZero):
        """
        Finds connections for a long island convex marginal point.

        Args:
            long_convex_marginal_point (int): The long island convex marginal point.
            point_connect_points_dict_gyri_parts (dict): A dictionary mapping points to their connected points.
            curv_data_delete_thicknessZero (dict): A dictionary mapping points to their curvature data.

        Returns:
            list: A list containing the connection dictionaries, the list of current outer points, and a flag indicating if the end is reached.
        """
        from_marginal_to_skelenton_father_sons_dict = defaultdict(list)
        from_skelenton_to_marginal_son_fathers_dict = defaultdict(list)
        connection_dicts_list = list()
        reach_end = 0
        current_outer_points_list = list()
        current_outer_points_list.append(long_convex_marginal_point)
        while not reach_end:
            next_outer_points_list = list()
            for point in current_outer_points_list:
                neighbors = [neighbor for neighbor in point_connect_points_dict_gyri_parts[point] if
                            neighbor not in from_marginal_to_skelenton_father_sons_dict.keys() and neighbor not in current_outer_points_list and
                            curv_data_delete_thicknessZero[neighbor] < 0]
                if point in from_marginal_to_skelenton_father_sons_dict.keys():
                    print('from_marginal_to_skelenton_father_sons_dict error!')
                else:
                    from_marginal_to_skelenton_father_sons_dict[point] = neighbors
                for neighbor in neighbors:
                    if neighbor not in from_skelenton_to_marginal_son_fathers_dict.keys():
                        from_skelenton_to_marginal_son_fathers_dict[neighbor] = [point]
                    else:
                        from_skelenton_to_marginal_son_fathers_dict[neighbor].append(point)

                next_outer_points_list = next_outer_points_list + neighbors

            if len(next_outer_points_list):
                current_outer_points_list = set(next_outer_points_list)
            else:
                connection_dicts_list.append(from_marginal_to_skelenton_father_sons_dict)
                connection_dicts_list.append(from_skelenton_to_marginal_son_fathers_dict)
                connection_dicts_list.append(current_outer_points_list)
                reach_end = 1
        return connection_dicts_list

    @classmethod
    def sort_the_long_path_convex_point(cls, long_path_convex_points_list, convex_marginal_points_length_dict):
        """
        Sorts the long path convex points based on their path length.

        Args:
            long_path_convex_points_list (list): A list of long path convex points.
            convex_marginal_points_length_dict (dict): A dictionary containing the path lengths of convex marginal points.

        Returns:
            list: A sorted list of long path convex points based on their path length.
        """
        long_path_convex_points_pathLength_dict = dict()
        for long_convex_point in long_path_convex_points_list:
            long_path_convex_points_pathLength_dict[long_convex_point] = convex_marginal_points_length_dict[long_convex_point][0]
        long_path_convex_points_pathLength_dict_sorted = sorted(long_path_convex_points_pathLength_dict.items(), key=lambda x: x[1], reverse=True)
        long_path_convex_points_list_sorted = [item[0] for item in long_path_convex_points_pathLength_dict_sorted]
        return long_path_convex_points_list_sorted

    @classmethod
    def find_connections_for_long_convex_marginal_point(cls, long_convex_marginal_point, final_point_lines_dict, point_connect_points_dict_gyri_parts, curv_data_delete_thicknessZero):
        """
        Finds connections for a long convex marginal point.

        Args:
            long_convex_marginal_point (int): The long convex marginal point to find connections for.
            final_point_lines_dict (dict): A dictionary mapping points to lines.
            point_connect_points_dict_gyri_parts (dict): A dictionary mapping points to their connected points.
            curv_data_delete_thicknessZero (dict): A dictionary mapping points to their thickness values.

        Returns:
            list: A list containing two dictionaries. The first dictionary maps marginal points to their skelenton father and sons. The second dictionary maps skelenton points to their marginal son and fathers.
        """
        from_marginal_to_skelenton_father_sons_dict = defaultdict(list)
        from_skelenton_to_marginal_son_fathers_dict = defaultdict(list)
        connection_dicts_list = list()
        reach_skelenton = 0
        skelenton_point_set = set()
        current_outer_points_list = list()
        current_outer_points_list.append(long_convex_marginal_point)
        while not reach_skelenton:
            next_outer_points_list = list()
            for point in current_outer_points_list:
                neighbors = [neighbor for neighbor in point_connect_points_dict_gyri_parts[point] if
                            neighbor not in from_marginal_to_skelenton_father_sons_dict.keys() and neighbor not in current_outer_points_list and
                            curv_data_delete_thicknessZero[neighbor] < 0]
                if point in from_marginal_to_skelenton_father_sons_dict.keys():
                    print('from_marginal_to_skelenton_father_sons_dict error!')
                else:
                    from_marginal_to_skelenton_father_sons_dict[point] = neighbors
                for neighbor in neighbors:
                    if neighbor not in from_skelenton_to_marginal_son_fathers_dict.keys():
                        from_skelenton_to_marginal_son_fathers_dict[neighbor] = [point]
                    else:
                        from_skelenton_to_marginal_son_fathers_dict[neighbor].append(point)

                next_outer_points_list = next_outer_points_list + neighbors

            if len(next_outer_points_list):
                skelenton_point_set = skelenton_point_set | (
                            set(next_outer_points_list) & set(final_point_lines_dict.keys()))
                if len(skelenton_point_set) >= 6:
                    reach_skelenton = 1
                    connection_dicts_list.append(from_marginal_to_skelenton_father_sons_dict)
                    connection_dicts_list.append(from_skelenton_to_marginal_son_fathers_dict)

                else:
                    current_outer_points_list = set(next_outer_points_list)
            else:
                break
        return connection_dicts_list

    @classmethod   
    def build_point_id_graphIndex_mapping(cls, convex_point, from_skelenton_to_marginal_son_fathers_dict):
        """
        Builds a mapping between point IDs and graph indices.

        Args:
            convex_point (int): The convex point ID.
            from_skelenton_to_marginal_son_fathers_dict (dict): A dictionary mapping points to their corresponding marginal son fathers.

        Returns:
            tuple: A tuple containing two dictionaries:
                - pointId_graphIndex_dict: A dictionary mapping point IDs to graph indices.
                - graphIndex_pointId_dict: A dictionary mapping graph indices to point IDs.
        """
        index = 0
        pointId_graphIndex_dict = dict()
        graphIndex_pointId_dict = dict()
        pointId_graphIndex_dict[convex_point] = index
        graphIndex_pointId_dict[index] = convex_point
        for point in from_skelenton_to_marginal_son_fathers_dict.keys():
            index += 1
            pointId_graphIndex_dict[point] = index
            graphIndex_pointId_dict[index] = point
        return pointId_graphIndex_dict, graphIndex_pointId_dict

    @classmethod
    def build_graph_for_convex_point(cls, convex_point, id_graphIndex_dict, from_marginal_to_skelenton_father_sons_dict, curv_data_delete_thicknessZero):
        """
        Builds a graph for a convex point based on the given parameters.

        Args:
            convex_point (int): The convex point.
            id_graphIndex_dict (dict): A dictionary mapping point IDs to graph indices.
            from_marginal_to_skelenton_father_sons_dict (dict): A dictionary mapping father points to their son points.
            curv_data_delete_thicknessZero (numpy.ndarray): An array containing curvature data with zero thickness removed.

        Returns:
            numpy.ndarray: The constructed graph.

        """
        graph = np.full((len(id_graphIndex_dict.keys()), len(id_graphIndex_dict.keys())), 100000000, dtype='float')
        for father_point in from_marginal_to_skelenton_father_sons_dict.keys():
            for son_point in from_marginal_to_skelenton_father_sons_dict[father_point]:
                # weight = 1/(abs(curv_data_delete_thicknessZero[son_point]) + 0.000000001)
                weight = 1 / (pow(curv_data_delete_thicknessZero[son_point], 2) + 0.000000001)
                graph[id_graphIndex_dict[father_point], id_graphIndex_dict[son_point]] = weight
                graph[id_graphIndex_dict[son_point], id_graphIndex_dict[father_point]] = weight
        return graph

    @classmethod
    def create_shortest_path(cls, adj, start, end):
        """
        Finds the shortest path between two nodes in a graph.

        Parameters:
        - adj (numpy.ndarray): The adjacency matrix representing the graph.
        - start (int): The index of the starting node.
        - end (int): The index of the target node.

        Returns:
        - r_list_numpy_index (list): The list of nodes representing the shortest path.
        - length (float): The length of the shortest path.
        """
        G = nx.Graph(adj)
        r_list_numpy_index = nx.dijkstra_path(G, source=start, target=end)
        length = nx.dijkstra_path_length(G, source=start, target=end)
        return r_list_numpy_index, length

    @classmethod
    def determine_real_long_path_by_curv(cls, path, graphIndex_pointId_dict, curv_data_delete_thicknessZero):
        """
        Determines the real long path by curv.

        Args:
            path (list): The list of indices representing the path.
            graphIndex_pointId_dict (dict): A dictionary mapping graph indices to point IDs.
            curv_data_delete_thicknessZero (list): The list of curvature data with zero thickness removed.

        Returns:
            list: The list of curvature values corresponding to the given path.
        """
        curv_of_path = list()
        for index in path:
            point = graphIndex_pointId_dict[index]
            curv_of_path.append(curv_data_delete_thicknessZero[point])
        return curv_of_path

    @classmethod
    def determine_real_long_path(cls, path, graphIndex_pointId_dict, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
        """
        Determines if a given path is a real long path based on certain conditions.

        Args:
            path (list): The path to be checked.
            graphIndex_pointId_dict (dict): A dictionary mapping graph indices to point IDs.
            point_patchSize_dict_orig (dict): A dictionary mapping point IDs to patch sizes.
            curv_data_delete_thicknessZero (float): The threshold for deleting thickness zero curvature data.
            flat_threshold_for_convex_gyri (int): The threshold for determining if a path is flat.

        Returns:
            bool: True if the path is a real long path, False otherwise.
        """
        max_patchsize = 0
        for index in path:
            point = graphIndex_pointId_dict[index]
            if point_patchSize_dict_orig[point] > max_patchsize:
                max_patchsize = point_patchSize_dict_orig[point]
        if len(path) < 2 * max_patchsize:
            curv_of_path = cls.determine_real_long_path_by_curv(path, graphIndex_pointId_dict, curv_data_delete_thicknessZero)
            curv_of_path_array = np.stack(curv_of_path)
            if np.sum(curv_of_path_array[0:6] > - 0.1) >= flat_threshold_for_convex_gyri:
                return False
            else:
                return True
        else:
            return True

    @classmethod
    def build_main_skeleton_for_missing_island_gyri(cls, longest_island_convex_point, final_point_lines_dict, point_connect_points_dict_gyri_parts, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
        """
        Builds the main skeleton for a missing island of gyri.

        Args:
            longest_island_convex_point (int): The convex point of the longest island.
            final_point_lines_dict (dict): A dictionary containing the final point lines.
            point_connect_points_dict_gyri_parts (dict): A dictionary mapping points to their connected points in gyri parts.
            point_patchSize_dict_orig (dict): A dictionary mapping points to their patch sizes.
            curv_data_delete_thicknessZero (float): The threshold for deleting thickness zero curvature data.
            flat_threshold_for_convex_gyri (float): The threshold for determining flatness of convex gyri.

        Returns:
            tuple: A tuple containing the updated final point lines dictionary and the connection dictionaries list.
        """
        print('build main skeleton for missing island \t' + time.asctime(time.localtime(time.time())))
        current_island_convex_point = longest_island_convex_point
        connection_dicts_list = cls.find_connections_for_long_island_convex_marginal_point(current_island_convex_point,
                                                                                    point_connect_points_dict_gyri_parts,
                                                                                    curv_data_delete_thicknessZero)
        if connection_dicts_list:
            from_marginal_to_skelenton_father_sons_dict = connection_dicts_list[0]
            from_skelenton_to_marginal_son_fathers_dict = connection_dicts_list[1]
            pointId_graphIndex_dict, graphIndex_pointId_dict = cls.build_point_id_graphIndex_mapping(current_island_convex_point,
                                                                                                from_skelenton_to_marginal_son_fathers_dict)
            graph = cls.build_graph_for_convex_point(current_island_convex_point, pointId_graphIndex_dict,
                                                from_marginal_to_skelenton_father_sons_dict,
                                                curv_data_delete_thicknessZero)
            skelenton_point_list = connection_dicts_list[2]
            shortest_length = 100000000
            shortest_path = []
            for skelenton_point in skelenton_point_list:
                skelenton_graph_id = pointId_graphIndex_dict[skelenton_point]
                long_convex_marginal_point_id = pointId_graphIndex_dict[current_island_convex_point]
                path_list, length = cls.create_shortest_path(graph, long_convex_marginal_point_id, skelenton_graph_id)
                if length < shortest_length:
                    shortest_path = path_list
                    shortest_length = length
            if cls.determine_real_long_path(shortest_path, graphIndex_pointId_dict, point_patchSize_dict_orig,
                                        curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
                for i in range(len(shortest_path) - 1):
                    if graphIndex_pointId_dict[shortest_path[i]] not in final_point_lines_dict.keys():
                        final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i]]] = {graphIndex_pointId_dict[shortest_path[i + 1]]}
                    else:
                        final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i]]].add(graphIndex_pointId_dict[shortest_path[i + 1]])
                    if graphIndex_pointId_dict[shortest_path[i + 1]] not in final_point_lines_dict.keys():
                        final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i + 1]]] = {graphIndex_pointId_dict[shortest_path[i]]}
                    else:
                        final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i + 1]]].add(graphIndex_pointId_dict[shortest_path[i]])
        return final_point_lines_dict, connection_dicts_list

    @classmethod
    def create_connection_for_missing_gyri(cls, orig_surf_polydata, orig_sphere_polydata, final_point_lines_dict, original_sulc_data, curv_data_delete_thicknessZero, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix):
        """
        Creates connections for missing gyri.

        Args:
            orig_surf_polydata (PolyData): The original surface polydata.
            orig_sphere_polydata (PolyData): The original sphere polydata.
            final_point_lines_dict (dict): A dictionary containing the final point lines.
            original_sulc_data (array): The original sulc data.
            curv_data_delete_thicknessZero (array): The curv data with thickness zero deleted.
            length_thres_of_long_gyri (float): The length threshold of long gyri.
            neighbor_missing_path_smallest_step (float): The smallest step for neighbor missing path.
            flat_threshold_for_convex_gyri (float): The flat threshold for convex gyri.
            nearest_skeleton_num (int): The number of nearest skeleton.
            island_gyri_length_thres (float): The length threshold of island gyri.
            output_prefix (str): The output prefix.

        Returns:
            dict: A dictionary containing the final point lines.
        """
        # Method implementation goes here
        ...
    def create_connection_for_missing_gyri(cls, orig_surf_polydata, orig_sphere_polydata, final_point_lines_dict, original_sulc_data, curv_data_delete_thicknessZero, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix):
        point_connect_points_dict_gyri_parts = cls.get_connect_points_gyri_part(orig_sphere_polydata, original_sulc_data)
        point_neighbor_points_dict = cls.get_connect_points(orig_sphere_polydata)
        point_patchSize_dict_orig = cls.find_the_patchSize_of_gyri_point(original_sulc_data, point_neighbor_points_dict)
        Draw.draw_patchSize_colorful_file(orig_sphere_polydata, orig_surf_polydata, point_patchSize_dict_orig, output_prefix, 'orig')
        _, marginal_points_gyri, _ = Points.find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, original_sulc_data)
        ThreeHG.draw_3hinge_on_surf(orig_sphere_polydata, marginal_points_gyri, output_prefix + '_sphere_marginal.vtk')
        ThreeHG.draw_3hinge_on_surf(orig_surf_polydata, marginal_points_gyri, output_prefix + '_surf_marginal.vtk')
        all_long_path_convex_points_list, convex_marginal_points_length_dict, island_convex_marginal_points_length_dict = cls.find_convex_marginal_points_of_long_gyri(orig_sphere_polydata, final_point_lines_dict, point_neighbor_points_dict, point_connect_points_dict_gyri_parts, marginal_points_gyri, original_sulc_data, curv_data_delete_thicknessZero, point_patchSize_dict_orig, length_thres_of_long_gyri, nearest_skeleton_num, island_gyri_length_thres)
        ThreeHG.draw_3hinge_on_surf(orig_sphere_polydata, all_long_path_convex_points_list, output_prefix + '_sphere_convex_marginal.vtk')
        ThreeHG.draw_3hinge_on_surf(orig_surf_polydata, all_long_path_convex_points_list, output_prefix + '_surf_convex_marginal.vtk')
        long_path_convex_points_list = cls.select_convex_endpoint_for_each_missing_gyri(all_long_path_convex_points_list, convex_marginal_points_length_dict, curv_data_delete_thicknessZero, point_connect_points_dict_gyri_parts, neighbor_missing_path_smallest_step)
        ThreeHG.draw_3hinge_on_surf(orig_sphere_polydata, long_path_convex_points_list, output_prefix + '_sphere_convex_marginal_long.vtk')
        ThreeHG.draw_3hinge_on_surf(orig_surf_polydata, long_path_convex_points_list, output_prefix + '_surf_convex_marginal_long.vtk')
        long_path_convex_points_list_sorted = cls.sort_the_long_path_convex_point(long_path_convex_points_list, convex_marginal_points_length_dict)
        missing_connection_list, final_point_lines_dict = cls.build_missing_connection_recurrently(long_path_convex_points_list_sorted, final_point_lines_dict, point_connect_points_dict_gyri_parts, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri)
        Writer.write_skelenton_by_connectionPair(orig_sphere_polydata, missing_connection_list, output_prefix + '_sphere_missing_connection.vtk')
        Writer.write_skelenton_by_connectionPair(orig_surf_polydata, missing_connection_list, output_prefix + '_surf_missing_connection.vtk')

        long_path_island_convex_candicate_points_list = list(island_convex_marginal_points_length_dict.keys())
        while long_path_island_convex_candicate_points_list:
            long_path_island_convex_candicate_points_list_sorted = cls.sort_the_long_path_convex_point(long_path_island_convex_candicate_points_list,
                                                                                                island_convex_marginal_points_length_dict)
            current_longest_island_convex_point = long_path_island_convex_candicate_points_list_sorted[0]
            final_point_lines_dict, connection_dicts_list = cls.build_main_skeleton_for_missing_island_gyri(current_longest_island_convex_point,
                                                                                                        final_point_lines_dict,
                                                                                                        point_connect_points_dict_gyri_parts,
                                                                                                        point_patchSize_dict_orig,
                                                                                                        curv_data_delete_thicknessZero,
                                                                                                        flat_threshold_for_convex_gyri)
            from_marginal_to_skelenton_father_sons_dict = connection_dicts_list[0]
            from_skelenton_to_marginal_son_fathers_dict = connection_dicts_list[1]
            island_convex_marginal_point_list = [point for point in from_skelenton_to_marginal_son_fathers_dict.keys()
                                                if point in island_convex_marginal_points_length_dict.keys()
                                                and point not in final_point_lines_dict.keys()]
            all_long_path_island_convex_points_list, convex_marginal_island_points_length_dict, _ = cls.find_convex_marginal_points_of_long_gyri(
                orig_sphere_polydata, final_point_lines_dict, point_neighbor_points_dict,
                point_connect_points_dict_gyri_parts, island_convex_marginal_point_list, original_sulc_data,
                curv_data_delete_thicknessZero, point_patchSize_dict_orig, length_thres_of_long_gyri, nearest_skeleton_num,
                island_gyri_length_thres)
            long_path_island_convex_points_list = cls.select_convex_endpoint_for_each_missing_gyri(all_long_path_island_convex_points_list,
                                                                                            convex_marginal_island_points_length_dict,
                                                                                            curv_data_delete_thicknessZero,
                                                                                            point_connect_points_dict_gyri_parts,
                                                                                            neighbor_missing_path_smallest_step)
            long_path_island_convex_points_list_sorted = cls.sort_the_long_path_convex_point(long_path_island_convex_points_list,
                                                                                        convex_marginal_island_points_length_dict)
            _, final_point_lines_dict = cls.build_missing_connection_recurrently(long_path_island_convex_points_list_sorted,
                                                                            final_point_lines_dict,
                                                                            point_connect_points_dict_gyri_parts,
                                                                            point_patchSize_dict_orig,
                                                                            curv_data_delete_thicknessZero,
                                                                            flat_threshold_for_convex_gyri)
            finished_island_convex_points_list = [point for point in island_convex_marginal_points_length_dict.keys()
                                                if point in from_marginal_to_skelenton_father_sons_dict.keys()
                                                or point in from_skelenton_to_marginal_son_fathers_dict.keys()]
            for point in finished_island_convex_points_list:
                long_path_island_convex_candicate_points_list.remove(point)

        return final_point_lines_dict
    
    @classmethod
    def build_missing_connection_recurrently(cls, long_path_convex_points_list_sorted, final_point_lines_dict, point_connect_points_dict_gyri_parts, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
        """
        Builds missing connections between points in a recurrent manner.

        Args:
            long_path_convex_points_list_sorted (list): A list of long convex marginal points.
            final_point_lines_dict (dict): A dictionary containing the final point lines.
            point_connect_points_dict_gyri_parts (dict): A dictionary mapping points to their connected points in gyri parts.
            point_patchSize_dict_orig (dict): A dictionary mapping points to their patch sizes.
            curv_data_delete_thicknessZero (float): The threshold for deleting thickness zero curvature data.
            flat_threshold_for_convex_gyri (float): The threshold for determining flat convex gyri.

        Returns:
            tuple: A tuple containing the missing connection list and the updated final point lines dictionary.
        """
        missing_connection_list = list()
        for long_convex_marginal_point in long_path_convex_points_list_sorted:
            connection_dicts_list = cls.find_connections_for_long_convex_marginal_point(long_convex_marginal_point,
                                                                                        final_point_lines_dict,
                                                                                        point_connect_points_dict_gyri_parts,
                                                                                        curv_data_delete_thicknessZero)
            if connection_dicts_list:
                from_marginal_to_skelenton_father_sons_dict = connection_dicts_list[0]
                from_skelenton_to_marginal_son_fathers_dict = connection_dicts_list[1]
                pointId_graphIndex_dict, graphIndex_pointId_dict = cls.build_point_id_graphIndex_mapping(long_convex_marginal_point,
                                                                                                        from_skelenton_to_marginal_son_fathers_dict)
                graph = cls.build_graph_for_convex_point(long_convex_marginal_point, pointId_graphIndex_dict,
                                                        from_marginal_to_skelenton_father_sons_dict,
                                                        curv_data_delete_thicknessZero)
                skelenton_point_list = [point for point in from_skelenton_to_marginal_son_fathers_dict.keys() if
                                        point in final_point_lines_dict.keys()]
                shortest_length = 100000000
                shortest_path = []
                for skelenton_point in skelenton_point_list:
                    skelenton_graph_id = pointId_graphIndex_dict[skelenton_point]
                    long_convex_marginal_point_id = pointId_graphIndex_dict[long_convex_marginal_point]
                    path_list, length = cls.create_shortest_path(graph, long_convex_marginal_point_id, skelenton_graph_id)
                    if length < shortest_length:
                        shortest_path = path_list
                        shortest_length = length
                if shortest_path:
                    if cls.determine_real_long_path(shortest_path, graphIndex_pointId_dict, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
                        for i in range(len(shortest_path) - 1):
                            if [graphIndex_pointId_dict[shortest_path[i]], graphIndex_pointId_dict[shortest_path[i + 1]]] not in missing_connection_list and [graphIndex_pointId_dict[shortest_path[i + 1]], graphIndex_pointId_dict[shortest_path[i]]] not in missing_connection_list:
                                missing_connection_list.append([graphIndex_pointId_dict[shortest_path[i]], graphIndex_pointId_dict[shortest_path[i + 1]]])

                            if graphIndex_pointId_dict[shortest_path[i]] not in final_point_lines_dict.keys():
                                final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i]]] = {graphIndex_pointId_dict[shortest_path[i + 1]]}
                            else:
                                final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i]]].add(graphIndex_pointId_dict[shortest_path[i + 1]])
                            if graphIndex_pointId_dict[shortest_path[i + 1]] not in final_point_lines_dict.keys():
                                final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i + 1]]] = {graphIndex_pointId_dict[shortest_path[i]]}
                            else:
                                final_point_lines_dict[graphIndex_pointId_dict[shortest_path[i + 1]]].add(graphIndex_pointId_dict[shortest_path[i]])
        return missing_connection_list, final_point_lines_dict
    
    @classmethod
    def find_skelenton_missing(cls, orig_sphere_polydata, orig_surf_polydata, skeleton_polydata, curv_data_delete_thicknessZero, original_sulc_data, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix):
        """
        Finds missing gyri in the skeleton.

        Args:
            orig_sphere_polydata: The original sphere polydata.
            orig_surf_polydata: The original surface polydata.
            skeleton_polydata: The skeleton polydata.
            curv_data_delete_thicknessZero: The curvature data with thickness zero deleted.
            original_sulc_data: The original sulc data.
            length_thres_of_long_gyri: The length threshold of long gyri.
            neighbor_missing_path_smallest_step: The smallest step for finding missing paths.
            flat_threshold_for_convex_gyri: The flat threshold for convex gyri.
            nearest_skeleton_num: The number of nearest skeleton points.
            island_gyri_length_thres: The length threshold of island gyri.
            output_prefix: The output prefix.

        Returns:
            None
        """
        print('================= build skeleton:\t' + time.asctime(time.localtime(time.time())) + '=======================')
        final_connection_list = cls.read_connection_of_skelenton_file(skeleton_polydata)
        final_point_lines_dict = cls.create_tree_connection_dict(final_connection_list)
        print('find missing gyri  \t' + time.asctime(time.localtime(time.time())))
        final_point_lines_dict = cls.create_connection_for_missing_gyri(orig_surf_polydata, orig_sphere_polydata,
                                                                    final_point_lines_dict, original_sulc_data,
                                                                    curv_data_delete_thicknessZero,
                                                                    length_thres_of_long_gyri,
                                                                    neighbor_missing_path_smallest_step,
                                                                    flat_threshold_for_convex_gyri, nearest_skeleton_num,
                                                                    island_gyri_length_thres, output_prefix)
        all_connection_list = cls.create_connectingPair(final_point_lines_dict)
        Writer.write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, all_connection_list, output_prefix + '_sphere_skelenton_allpoints_final.vtk')
        Writer.write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata, all_connection_list, output_prefix + '_surf_skelenton_allpoints_final.vtk')

        all_point_lines_dict = cls.create_tree_connection_dict(all_connection_list)

        print('find and draw 3hinge' + time.asctime(time.localtime(time.time())))
        hinge3_list = ThreeHG.create_3hinge(all_point_lines_dict)
        hinge3_txt = open(output_prefix + '_3hinge_ids.txt', "w")
        for hinge in hinge3_list:
            hinge3_txt.write(str(hinge) + "\n")
        hinge3_txt.close()
        ThreeHG.draw_3hinge_on_surf(orig_sphere_polydata, hinge3_list, output_prefix + '_sphere_3hinge_vertex.vtk')
        ThreeHG.draw_3hinge_on_surf(orig_surf_polydata, hinge3_list, output_prefix + '_surf_3hinge_vertex.vtk')

