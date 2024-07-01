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


class Tree:

    @classmethod
    def find_the_max_patchSize_point(cls, unconnected_points_list, point_patchSize_dict_updated):
        """
        Finds the point with the maximum patch size from a list of unconnected points.

        Args:
            unconnected_points_list (list): A list of unconnected points.
            point_patchSize_dict_updated (dict): A dictionary mapping points to their patch sizes.

        Returns:
            The point with the maximum patch size.

        """
        max = -1
        for point in unconnected_points_list:
            if point_patchSize_dict_updated[point] > max:
                max = point_patchSize_dict_updated[point]
                max_patchSize_point = point
        return max_patchSize_point

    @classmethod
    def create_tree(cls, orig_sphere_polydata, orig_surf_polydata, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, thin_sulc_data, output_prefix):
        """
        Creates a tree structure based on the given inputs.

        Args:
            orig_sphere_polydata: The original sphere polydata.
            orig_surf_polydata: The original surface polydata.
            point_patchSize_dict_updated: A dictionary mapping points to their patch sizes.
            point_connect_points_dict_thin_gyri_parts: A dictionary mapping points to their connected points in thin gyri parts.
            thin_sulc_data: The thin sulc data.
            output_prefix: The output prefix for the generated files.

        Returns:
            A tuple containing the connected lines list and the father dictionary.
        """
        print('================= create tree:\t' + time.asctime(time.localtime(time.time())) + '=========================')
        point_num = orig_sphere_polydata.GetNumberOfPoints()

        connected_points_list = list()
        connected_lines_list = list()
        connected_candidate_dict = defaultdict(list)
        unconnected_points_list = list()
        father_dict = dict()
        gyri_connection_count = cls.create_gyri_connection_count(point_connect_points_dict_thin_gyri_parts)

        for point in range(point_num):
            if thin_sulc_data[point] < 0:
                unconnected_points_list.append(point)

        debug_counter = 0

        while len(unconnected_points_list) != 0:
            print(' remaining points:\t', len(unconnected_points_list))
            if not connected_candidate_dict:
                # pdb.set_trace()
                max_patchSize_point = cls.find_the_max_patchSize_point(unconnected_points_list, point_patchSize_dict_updated)
                connected_points_list.append(max_patchSize_point)
                unconnected_points_list.remove(max_patchSize_point)
                father_dict[max_patchSize_point] = max_patchSize_point
            else:
                connected_candidate_dict = {}

            available_connected_points_list = [point for point in connected_points_list if gyri_connection_count[point] > 0]
            for point in available_connected_points_list:
                # if gyri_connection_count[point]:
                neighbors = [neighbor for neighbor in point_connect_points_dict_thin_gyri_parts[point] if
                            neighbor not in connected_points_list]
                for neighbor in neighbors:
                    if neighbor not in connected_candidate_dict.keys():
                        connected_candidate_dict[neighbor] = [[neighbor, point]]
                    else:
                        connected_candidate_dict[neighbor].append([neighbor, point])

            if connected_candidate_dict:
                candidate = Points.find_the_max_patchSize_candidate_connection(point_patchSize_dict_updated, connected_candidate_dict)
                connected_points_list.append(candidate[0])
                unconnected_points_list.remove(candidate[0])
                connected_lines_list.append(candidate)
                # pdb.set_trace()
                for connection_of_candidate in point_connect_points_dict_thin_gyri_parts[candidate[0]]:
                    gyri_connection_count[connection_of_candidate] = gyri_connection_count[connection_of_candidate] - 1
                if candidate[0] not in father_dict.keys():
                    father_dict[candidate[0]] = candidate[1]
                else:
                    print('reconnecting the existing point!')
                    exit()
                debug_counter += 1
                # if debug_counter % 1000 == 0:
                #     write_skelenton_by_connectionPair(orig_sphere_polydata, connected_lines_list, output_prefix + '_sphere_skeleton_round(' + str(debug_counter) + ').vtk')
        for connection in connected_lines_list:
            if [connection[1], connection[0]] in connected_lines_list:
                print('connection already in connected_lines_list')
                exit()

        Writer.write_skelenton_by_connectionPair(orig_sphere_polydata, connected_lines_list, output_prefix + '_sphere_initial_tree.vtk')
        Writer.write_skelenton_by_connectionPair(orig_surf_polydata, connected_lines_list, output_prefix + '_surf_initial_tree.vtk')

        return connected_lines_list, father_dict

    @classmethod
    def step_n_father(cls, point, n, father_dict):
        """
        Returns a list of the ancestors of a given point up to a specified number of steps.

        Args:
            point (int): The starting point.
            n (int): The number of steps to go back.
            father_dict (dict): A dictionary mapping each point to its parent.

        Returns:
            list: A list of the ancestors of the given point up to the specified number of steps.
        """
        point_father_list = list()
        current_father = father_dict[point]
        for step in range(n):
            point_father_list.append(current_father)
            current_father = father_dict[current_father]
        return point_father_list
    
    @classmethod
    def trim_allRound1_long(cls, point_lines_dict, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, father_dict, inner_gyri_points):
        '''Deletes round-1 short branches and long branches from the given point_lines_dict.

        Args:
            point_lines_dict (dict): A dictionary containing the points and their connecting points.
            point_patchSize_dict_updated (dict): A dictionary containing the patch sizes of the points.
            point_connect_points_dict_thin_gyri_parts (dict): A dictionary containing the connecting points of the thin gyri parts.
            father_dict (dict): A dictionary containing the father points of the points.
            inner_gyri_points (list): A list of inner gyri points.

        Returns:
            tuple: A tuple containing the updated point_lines_dict and the list of deleted round-1 points.
        '''
        # delete round-1 short branches
        deleted_round1_points = list()
        for point in point_lines_dict.keys():
            connecting_points = list(point_lines_dict[point])
            long_edge_num = 0
            short_edge_max_patchsize = 0
            for connecting_point in connecting_points:
                if len(point_lines_dict[connecting_point]) > 1:
                    long_edge_num += 1
                elif len(point_lines_dict[connecting_point]) == 1:
                    if point_patchSize_dict_updated[connecting_point] > short_edge_max_patchsize:
                        short_edge_max_patchsize = point_patchSize_dict_updated[connecting_point]
                        short_edge_candidate = connecting_point
            current_deleted_round1_points_list = list()
            if long_edge_num < len(connecting_points):
                for connecting_point in connecting_points:
                    if len(point_lines_dict[connecting_point]) == 1 and connecting_point != short_edge_candidate:
                        point_lines_dict[point].remove(connecting_point)
                        point_lines_dict[connecting_point].remove(point)
                        current_deleted_round1_points_list.append(connecting_point)
                if long_edge_num > 1:
                    point_lines_dict[point].remove(short_edge_candidate)
                    point_lines_dict[short_edge_candidate].remove(point)
                    current_deleted_round1_points_list.append(short_edge_candidate)
            if long_edge_num == 1:
                deleted_round1_points = list(set(deleted_round1_points + current_deleted_round1_points_list))

        # delete long branches
        branch_endpoint_list = list()
        for endpoint in point_lines_dict.keys():
            if len(point_lines_dict[endpoint]) == 1:
                branch_endpoint_list.append(endpoint)
        for endpoint in branch_endpoint_list:
            step_n_father_list = cls.step_n_father(endpoint, 6, father_dict)
            first_round_referencePoint_list_of_endpoint = [referencePoint_of_endpoint for referencePoint_of_endpoint in
                                                        point_connect_points_dict_thin_gyri_parts[endpoint]
                                                        if referencePoint_of_endpoint in point_lines_dict.keys()
                                                        and len(point_lines_dict[referencePoint_of_endpoint]) >= 2
                                                        and (referencePoint_of_endpoint not in step_n_father_list
                                                                or (referencePoint_of_endpoint in step_n_father_list
                                                                    and max(cls.get_connection_degree_of_step_n_father(endpoint, referencePoint_of_endpoint, father_dict, point_lines_dict)) > 2))]
            if len(first_round_referencePoint_list_of_endpoint) > 0:
                referencePoint_list_of_endpoint = list(set(first_round_referencePoint_list_of_endpoint))
            else:
                inner_neighbor_list = [inner_neighbor for inner_neighbor in point_connect_points_dict_thin_gyri_parts[endpoint]
                                    if inner_neighbor in inner_gyri_points]
                second_round_referencePoint_list_of_endpoint = list()
                for inner_neighbor in inner_neighbor_list:
                    referencePoint_list = [referencePoint for referencePoint in
                                        point_connect_points_dict_thin_gyri_parts[inner_neighbor]
                                        if referencePoint in point_lines_dict.keys()
                                        and len(point_lines_dict[referencePoint]) >= 2
                                        and (referencePoint not in step_n_father_list
                                                or (referencePoint in step_n_father_list
                                                    and max(cls.get_connection_degree_of_step_n_father(endpoint, referencePoint, father_dict, point_lines_dict)) > 2))]
                    second_round_referencePoint_list_of_endpoint = second_round_referencePoint_list_of_endpoint + referencePoint_list
                referencePoint_list_of_endpoint = list(set(second_round_referencePoint_list_of_endpoint))

            same_branch = 0
            for referencePoint_of_endpoint in referencePoint_list_of_endpoint:
                if father_dict[referencePoint_of_endpoint] in point_connect_points_dict_thin_gyri_parts[father_dict[endpoint]] or father_dict[referencePoint_of_endpoint] in point_connect_points_dict_thin_gyri_parts[father_dict[father_dict[endpoint]]]:
                    same_branch = 1
                    break
                else:
                    endpoint_father_list = cls.step_n_father(endpoint, 12, father_dict)
                    referencePoint_father_list = cls.step_n_father(referencePoint_of_endpoint, 12, father_dict)
                    if list(set(endpoint_father_list) & set(referencePoint_father_list)):
                        same_branch = 1
                        break

            if same_branch:
                current_endpoint = endpoint
                while len(point_lines_dict[current_endpoint]) == 1:
                    next_endpoint = list(point_lines_dict[current_endpoint])[0]
                    point_lines_dict[current_endpoint].remove(next_endpoint)
                    point_lines_dict[next_endpoint].remove(current_endpoint)
                    current_endpoint = next_endpoint

        return point_lines_dict, deleted_round1_points

    @classmethod
    def create_connectingPair(cls, point_lines_dict):
        """
        Creates a list of connecting pairs from a dictionary of points and their connecting points.

        Args:
            point_lines_dict (dict): A dictionary where the keys are points and the values are lists of connecting points.

        Returns:
            list: A list of connecting pairs, where each pair is represented as a list [point, connecting_point].

        """
        final_connection_list = list()
        for point in point_lines_dict.keys():
            for connecting_point in point_lines_dict[point]:
                if [point, connecting_point] not in final_connection_list and [connecting_point, point] not in final_connection_list:
                    final_connection_list.append([point, connecting_point])
        return final_connection_list

    @classmethod
    def connect_break_in_circle(cls, point_lines_dict, point_connect_points_dict_thin_gyri_parts, deleted_round1_points, father_dict):
        """
        Connects break in a circle by adding missing connections between points belonging to the same branch.

        Parameters:
        - point_lines_dict (dict): A dictionary mapping points to a set of connected lines.
        - point_connect_points_dict_thin_gyri_parts (dict): A dictionary mapping points to a list of neighboring points.
        - deleted_round1_points (list): A list of points that were deleted in the first round.
        - father_dict (dict): A dictionary mapping points to their respective fathers.

        Returns:
        - point_lines_dict (dict): The updated dictionary mapping points to a set of connected lines.
        """
        for point in point_lines_dict.keys():
            if len(point_lines_dict[point]) == 1:
                find_neighbor_endpoint = 0
                for neighbor in point_connect_points_dict_thin_gyri_parts[point]:
                    if neighbor in point_lines_dict.keys() and len(point_lines_dict[neighbor]) == 1:
                        # pdb.set_trace()
                        if not list(set(cls.step_n_father(point, 10, father_dict)) & set(cls.step_n_father(neighbor, 10, father_dict))):
                            point_lines_dict[point].add(neighbor)
                            point_lines_dict[neighbor].add(point)
                            find_neighbor_endpoint = 1
                        else:
                            print('Case1-neighbor: connect two points belonging to the same branch:', point, neighbor)

                if find_neighbor_endpoint == 0:
                    # if point == 109497:
                    #     pdb.set_trace()
                    for neighbor in point_connect_points_dict_thin_gyri_parts[point]:
                        if neighbor in deleted_round1_points:
                            # pdb.set_trace()
                            for second_neighbor in point_connect_points_dict_thin_gyri_parts[neighbor]:
                                if len(point_lines_dict[second_neighbor]) == 1 and second_neighbor != point:
                                    if not list(set(cls.step_n_father(point, 10, father_dict)) & set(cls.step_n_father(second_neighbor, 10, father_dict))):
                                        point_lines_dict[point].add(neighbor)
                                        point_lines_dict[neighbor] = {point}
                                        point_lines_dict[neighbor].add(second_neighbor)
                                        point_lines_dict[second_neighbor].add(neighbor)
                                        find_neighbor_endpoint = 1
                                        break
                                    else:
                                        print('Case2-deleted_round1: connect two points belonging to the same branch:', point, neighbor, second_neighbor)
                        if find_neighbor_endpoint:
                            break

                if find_neighbor_endpoint == 0:
                    father = father_dict[point]
                    for father_neighbor in point_connect_points_dict_thin_gyri_parts[father]:
                        if father_neighbor != point and len(point_lines_dict[father_neighbor]) == 1:
                            if not list(set(cls.step_n_father(father, 10, father_dict)) & set(cls.step_n_father(father_neighbor, 10, father_dict))):
                                point_lines_dict[father].add(father_neighbor)
                                point_lines_dict[father_neighbor].add(father)
                                point_lines_dict[father].remove(point)
                                point_lines_dict[point].remove(father)
                                find_neighbor_endpoint = 1
                                break
                            else:
                                print('Case3-father: connect two points belonging to the same branch:', father, father_neighbor)

                if find_neighbor_endpoint == 0:
                    neighbor_list = [neighbor for neighbor in point_connect_points_dict_thin_gyri_parts[point]
                                    if neighbor != father_dict[point]]
                    neighbor_second_neighbor_pairs_list = list()
                    for neighbor in neighbor_list:
                        if len(point_lines_dict[neighbor]) > 0:
                            # pass
                            print('two endpoints neighbored!', point, neighbor)
                        else:
                            neighbor_second_neighbor_pairs = [[neighbor, second_neighbor] for second_neighbor in
                                                            point_connect_points_dict_thin_gyri_parts[neighbor]
                                                            if second_neighbor in point_lines_dict.keys()
                                                            and len(point_lines_dict[second_neighbor]) == 1
                                                            and second_neighbor not in [point, father_dict[point], father_dict[father_dict[point]]]]
                            neighbor_second_neighbor_pairs_list = neighbor_second_neighbor_pairs_list + neighbor_second_neighbor_pairs
                    for neighbor_second_neighbor_pair in neighbor_second_neighbor_pairs_list:
                        neighbor = neighbor_second_neighbor_pair[0]
                        second_neighbor = neighbor_second_neighbor_pair[1]
                        if not list(set(cls.step_n_father(point, 10, father_dict)) & set(cls.step_n_father(second_neighbor, 10, father_dict))):
                            point_lines_dict[point].add(neighbor)
                            point_lines_dict[neighbor] = {point}
                            point_lines_dict[neighbor].add(second_neighbor)
                            point_lines_dict[second_neighbor].add(neighbor)
                            find_neighbor_endpoint = 1
                            break
                        else:
                            print('Case4-round2:connect two points belonging to the same branch:', point, neighbor, second_neighbor)

                if find_neighbor_endpoint == 0:
                    first_deleted_round1_neighbor_list = [neighbor for neighbor in point_connect_points_dict_thin_gyri_parts[point]
                                                        if neighbor in deleted_round1_points]
                    if len(first_deleted_round1_neighbor_list):
                        for first_deleted_round1_neighbor in first_deleted_round1_neighbor_list:
                            second_deleted_round1_neighbor_list = [second_neighbor for second_neighbor in
                                                                point_connect_points_dict_thin_gyri_parts[first_deleted_round1_neighbor]
                                                                if second_neighbor in deleted_round1_points
                                                                and second_neighbor not in first_deleted_round1_neighbor_list]
                            for second_deleted_round1_neighbor in second_deleted_round1_neighbor_list:
                                for third_neighbor in point_connect_points_dict_thin_gyri_parts[second_deleted_round1_neighbor]:
                                    if len(point_lines_dict[third_neighbor]) == 1:
                                        if not list(set(cls.step_n_father(point, 10, father_dict)) & set(cls.step_n_father(third_neighbor, 10, father_dict))):
                                            point_lines_dict[point].add(first_deleted_round1_neighbor)
                                            point_lines_dict[first_deleted_round1_neighbor].add(point)
                                            point_lines_dict[first_deleted_round1_neighbor].add(second_deleted_round1_neighbor)
                                            point_lines_dict[second_deleted_round1_neighbor].add(first_deleted_round1_neighbor)
                                            point_lines_dict[second_deleted_round1_neighbor].add(third_neighbor)
                                            point_lines_dict[third_neighbor].add(second_deleted_round1_neighbor)
                                            find_neighbor_endpoint = 1
                                            break
                                        else:
                                            print('Case5-deleted_round1: connect two points belonging to the same branch:',
                                                point, first_deleted_round1_neighbor, second_deleted_round1_neighbor, third_neighbor)
                                if find_neighbor_endpoint:
                                    break
                            if find_neighbor_endpoint:
                                break

                if find_neighbor_endpoint == 0:
                    print('remaining endpoint:', point)
        return point_lines_dict

    @classmethod
    def clear_empty_point_in_dict(cls, point_lines_dict):
        """
        Removes empty point lines from a dictionary.

        Args:
            point_lines_dict (dict): A dictionary containing point lines.

        Returns:
            dict: A dictionary with empty point lines removed.
        """
        clear_point_lines_dict = defaultdict(set)
        for point in point_lines_dict.keys():
            if len(point_lines_dict[point]):
                clear_point_lines_dict[point] = point_lines_dict[point]
        return clear_point_lines_dict

    @classmethod
    def find_skelenton(cls, orig_sphere_polydata, orig_surf_polydata, point_patchSize_dict_updated, curv_data_delete_thicknessZero, original_sulc_data, thin_sulc_data, point_neighbor_points_dict, point_connect_points_dict_thin_gyri_parts, connected_lines_list, father_dict, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix):
        """
        Builds the skeleton by performing various operations such as trimming, connecting breaks, finding missing gyri, and identifying 3-hinge points.

        Args:
            orig_sphere_polydata: The original sphere polydata.
            orig_surf_polydata: The original surface polydata.
            point_patchSize_dict_updated: A dictionary mapping points to their patch sizes.
            curv_data_delete_thicknessZero: The curv data after deleting thickness zero.
            original_sulc_data: The original sulc data.
            thin_sulc_data: The thin sulc data.
            point_neighbor_points_dict: A dictionary mapping points to their neighbor points.
            point_connect_points_dict_thin_gyri_parts: A dictionary mapping points to their connected points in thin gyri parts.
            connected_lines_list: A list of connected lines.
            father_dict: A dictionary mapping points to their father points.
            length_thres_of_long_gyri: The length threshold of long gyri.
            neighbor_missing_path_smallest_step: The smallest step for finding missing paths between neighbors.
            flat_threshold_for_convex_gyri: The flat threshold for convex gyri.
            nearest_skeleton_num: The number of nearest skeleton points.
            island_gyri_length_thres: The length threshold for island gyri.
            output_prefix: The output prefix for the generated files.

        Returns:
            None
        """
        print('================= build skeleton:\t' + time.asctime(time.localtime(time.time())) + '=======================')
        marginal_points, marginal_points_gyri, marginal_points_sulc = Points.find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, thin_sulc_data)

        print('begin trimming  \t' + time.asctime(time.localtime(time.time())))
        tree_point_lines_dict = cls.create_tree_connection_dict(connected_lines_list)
        inner_gyri_points = [gyri_point for gyri_point in tree_point_lines_dict if gyri_point not in marginal_points_gyri]
        initial_skelenton_point_lines_dict, deleted_round1_points = cls.trim_allRound1_long(tree_point_lines_dict, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, father_dict, inner_gyri_points)
        initial_skelenton_connection_list = cls.create_connectingPair(initial_skelenton_point_lines_dict)
        print('draw initial skelenton  \t' + time.asctime(time.localtime(time.time())))
        Writer.write_skelenton_by_connectionPair(orig_sphere_polydata, initial_skelenton_connection_list, output_prefix + '_sphere_initial_skelenton.vtk')
        Writer.write_skelenton_by_connectionPair(orig_surf_polydata, initial_skelenton_connection_list, output_prefix + '_surf_initial_skelenton.vtk')

        print('connecting breaks in skelenton  \t' + time.asctime(time.localtime(time.time())))
        final_point_lines_dict = cls.connect_break_in_circle(initial_skelenton_point_lines_dict, point_connect_points_dict_thin_gyri_parts, deleted_round1_points, father_dict)
        final_connection_list = cls.create_connectingPair(final_point_lines_dict)
        Writer.write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, final_connection_list, output_prefix + '_sphere_skelenton_connect_break_allpoints.vtk')
        Writer.write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata, final_connection_list, output_prefix + '_surf_skelenton_connect_break_allpoints.vtk')

        final_point_lines_dict = cls.clear_empty_point_in_dict(final_point_lines_dict)

        print('find missing gyri  \t' + time.asctime(time.localtime(time.time())))
        final_point_lines_dict = cls.create_connection_for_missing_gyri(orig_surf_polydata, orig_sphere_polydata, final_point_lines_dict, original_sulc_data, curv_data_delete_thicknessZero, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix)
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


"""
我们看到了一个名为[`find_marginal_point`]的方法的定义。这个方法用于根据给定的条件找到边缘点。它接受一个点列表、一个点到其邻居点的映射字典和一个包含每个点的凹凸数据的数组作为参数。它返回一个包含三个列表的元组，分别是所有边缘点、凸部分的边缘点和凹部分的边缘点。

接下来，我们看到了一个名为[`trim_allRound1_long`]的方法的定义。这个方法用于删除骨架中的一些分支。它接受一个点到连接点的字典、一个点到补丁大小的字典、一个连接点到连接点的字典、一个点到父节点的字典和一个内部凸点列表作为参数。它返回更新后的点到连接点的字典和删除的一轮分支的列表。

然后，我们看到了一个名为[`create_connectingPair`]的方法的定义。这个方法用于从点到连接点的字典创建一个连接对的列表。它遍历字典中的每个点，并将每个连接点与该点组成的连接对添加到列表中。

最后，我们看到了一个名为[`write_skelenton_by_connectionPair`]的方法的定义。这个方法用于将连接对的列表写入文件。它接受原始球面和连接线列表作为参数，并将它们写入指定的输出文件。
"""