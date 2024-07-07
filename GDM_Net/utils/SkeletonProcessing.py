'''
Author: HenryVarro666 1504517223@qq.com
Date: 1969-12-31 19:00:00
LastEditors: HenryVarro666 1504517223@qq.com
LastEditTime: 2024-07-07 11:15:26
FilePath: /DensityMap-GyralNet/GDM_Net/utils/skeleton.py
'''
import nibabel.freesurfer.io as io
import numpy as np
import pyvista
import vtk
import os
from collections import defaultdict
import time
import shutil

from utils.data_io import read_vtk_file, clear_output_folder
from utils.draw import write_featured_sphere_from_variable, write_featured_sphere_from_variable_single
# from utils.draw import featured_sphere
from utils.draw import get_connect_points, draw_patchSize_colorful_file
from utils.augmentation import delete_isolated_point, find_marginal_point
from utils.tree import get_connect_points_gyri_part, create_tree

import networkx as nx

class Find_skelenton:
    def create_tree_connection_dict(connected_lines_list):
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

    def step_n_father(point, n, father_dict):
        point_father_list = list()
        current_father = father_dict[point]
        for step in range(n):
            point_father_list.append(current_father)
            current_father = father_dict[current_father]
        return point_father_list

    def get_connection_degree_of_step_n_father(start_point, end_father, father_dict, point_lines_dict):
        connection_degrees = list()
        father = father_dict[start_point]
        while father != end_father:
            degree = len(point_lines_dict[father])
            connection_degrees.append(degree)
            father = father_dict[father]
        degree = len(point_lines_dict[father])
        connection_degrees.append(degree)
        return connection_degrees

    def trim_allRound1_long(point_lines_dict, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, father_dict, inner_gyri_points):
        '''delete round-1 short branches'''
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

        '''delete long branches'''
        branch_endpoint_list = list()
        for endpoint in point_lines_dict.keys():
            if len(point_lines_dict[endpoint]) == 1:
                branch_endpoint_list.append(endpoint)
        for endpoint in branch_endpoint_list:
            # if endpoint in [45287, 99484, 119924]:
            #     pdb.set_trace()
            step_n_father_list = Find_skelenton.step_n_father(endpoint, 6, father_dict)
            first_round_referencePoint_list_of_endpoint = [referencePoint_of_endpoint for referencePoint_of_endpoint in
                                                        point_connect_points_dict_thin_gyri_parts[endpoint]
                                                        if referencePoint_of_endpoint in point_lines_dict.keys()
                                                        and len(point_lines_dict[referencePoint_of_endpoint]) >= 2
                                                        and (referencePoint_of_endpoint not in step_n_father_list
                                                                or (referencePoint_of_endpoint in step_n_father_list
                                                                    and max(Find_skelenton.get_connection_degree_of_step_n_father(endpoint, referencePoint_of_endpoint, father_dict, point_lines_dict)) > 2))]
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
                                                    and max(Find_skelenton.get_connection_degree_of_step_n_father(endpoint, referencePoint, father_dict, point_lines_dict)) > 2))]
                    second_round_referencePoint_list_of_endpoint = second_round_referencePoint_list_of_endpoint + referencePoint_list
                referencePoint_list_of_endpoint = list(set(second_round_referencePoint_list_of_endpoint))

            same_branch = 0
            for referencePoint_of_endpoint in referencePoint_list_of_endpoint:
                if father_dict[referencePoint_of_endpoint] in point_connect_points_dict_thin_gyri_parts[father_dict[endpoint]] or father_dict[referencePoint_of_endpoint] in point_connect_points_dict_thin_gyri_parts[father_dict[father_dict[endpoint]]]:
                    same_branch = 1
                    break
                else:
                    endpoint_father_list = Find_skelenton.step_n_father(endpoint, 12, father_dict)
                    referencePoint_father_list = Find_skelenton.step_n_father(referencePoint_of_endpoint, 12, father_dict)
                    if list(set(endpoint_father_list) & set(referencePoint_father_list)):
                        same_branch = 1
                        break

            if same_branch:
                current_endpoint = endpoint
                while len(point_lines_dict[current_endpoint]) == 1:
                    # pdb.set_trace()
                    next_endpoint = list(point_lines_dict[current_endpoint])[0]
                    point_lines_dict[current_endpoint].remove(next_endpoint)
                    point_lines_dict[next_endpoint].remove(current_endpoint)
                    current_endpoint = next_endpoint

        return point_lines_dict, deleted_round1_points

    def create_connectingPair(point_lines_dict):
        final_connection_list = list()
        for point in point_lines_dict.keys():
            for connecting_point in point_lines_dict[point]:
                if [point, connecting_point] not in final_connection_list and [connecting_point, point] not in final_connection_list:
                    final_connection_list.append([point, connecting_point])
        return final_connection_list

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

    def connect_break_in_circle(point_lines_dict, point_connect_points_dict_thin_gyri_parts, deleted_round1_points, father_dict):
        for point in point_lines_dict.keys():
            if len(point_lines_dict[point]) == 1:
                find_neighbor_endpoint = 0
                for neighbor in point_connect_points_dict_thin_gyri_parts[point]:
                    if neighbor in point_lines_dict.keys() and len(point_lines_dict[neighbor]) == 1:
                        # pdb.set_trace()
                        if not list(set(Find_skelenton.step_n_father(point, 10, father_dict)) & set(Find_skelenton.step_n_father(neighbor, 10, father_dict))):
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
                                    if not list(set(Find_skelenton.step_n_father(point, 10, father_dict)) & set(Find_skelenton.step_n_father(second_neighbor, 10, father_dict))):
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
                            if not list(set(Find_skelenton.step_n_father(father, 10, father_dict)) & set(Find_skelenton.step_n_father(father_neighbor, 10, father_dict))):
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
                        if not list(set(Find_skelenton.step_n_father(point, 10, father_dict)) & set(Find_skelenton.step_n_father(second_neighbor, 10, father_dict))):
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
                                        if not list(set(Find_skelenton.step_n_father(point, 10, father_dict)) & set(Find_skelenton.step_n_father(third_neighbor, 10, father_dict))):
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

    def write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, connected_lines_list, output):
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

    def clear_empty_point_in_dict(point_lines_dict):
        clear_point_lines_dict = defaultdict(set)
        for point in point_lines_dict.keys():
            if len(point_lines_dict[point]):
                clear_point_lines_dict[point] = point_lines_dict[point]
        return clear_point_lines_dict

    def find_the_patchSize_of_gyri_point(sulc_data, point_neighbor_points_dict):
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

    def find_round_n_neighbor(point, n, point_neighbor_points_dict):
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

    def select_convex_endpoint_for_each_missing_gyri(long_path_convex_points_list, 
                                                     convex_marginal_points_length_dict, 
                                                     curv_data_delete_thicknessZero, 
                                                     point_connect_points_dict_gyri_parts, 
                                                     neighbor_missing_path_smallest_step):
        print('select convex endpoint for each missing gyri \t' + time.asctime(time.localtime(time.time())))
        delete_list = list()
        for convex_point in long_path_convex_points_list:
            point_connection = [point for point in convex_marginal_points_length_dict[convex_point][1].keys() if point in long_path_convex_points_list]
            current_point = point_connection[0]
            max_path = convex_marginal_points_length_dict[current_point][0]
            step_n_neighbor_list, _ = Find_skelenton.find_round_n_neighbor(convex_point, neighbor_missing_path_smallest_step, point_connect_points_dict_gyri_parts)
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

    def sort_the_long_path_convex_point(long_path_convex_points_list, convex_marginal_points_length_dict):
        long_path_convex_points_pathLength_dict = dict()
        for long_convex_point in long_path_convex_points_list:
            long_path_convex_points_pathLength_dict[long_convex_point] = convex_marginal_points_length_dict[long_convex_point][0]
        long_path_convex_points_pathLength_dict_sorted = sorted(long_path_convex_points_pathLength_dict.items(), key=lambda x: x[1], reverse=True)
        long_path_convex_points_list_sorted = [item[0] for item in long_path_convex_points_pathLength_dict_sorted]
        return long_path_convex_points_list_sorted

    def find_connections_for_long_convex_marginal_point(long_convex_marginal_point, 
                                                        final_point_lines_dict, 
                                                        point_connect_points_dict_gyri_parts, 
                                                        curv_data_delete_thicknessZero):
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

    def build_point_id_graphIndex_mapping(convex_point, from_skelenton_to_marginal_son_fathers_dict):
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

    def build_graph_for_convex_point(convex_point, id_graphIndex_dict, from_marginal_to_skelenton_father_sons_dict, curv_data_delete_thicknessZero):
        graph = np.full((len(id_graphIndex_dict.keys()), len(id_graphIndex_dict.keys())), 100000000, dtype='float')
        for father_point in from_marginal_to_skelenton_father_sons_dict.keys():
            for son_point in from_marginal_to_skelenton_father_sons_dict[father_point]:
                # weight = 1/(abs(curv_data_delete_thicknessZero[son_point]) + 0.000000001)
                weight = 1 / (pow(curv_data_delete_thicknessZero[son_point], 2) + 0.000000001)
                graph[id_graphIndex_dict[father_point], id_graphIndex_dict[son_point]] = weight
                graph[id_graphIndex_dict[son_point], id_graphIndex_dict[father_point]] = weight
        return graph
        
    def create_shortest_path(adj, start, end):
        # pdb.set_trace()
        # for i in range(adj.shape[0]):
        #     print(adj[i, :].min(), adj[:, i].min())
        G = nx.Graph(adj)
        r_list_numpy_index = nx.dijkstra_path(G, source=start, target=end)
        length = nx.dijkstra_path_length(G, source=start, target=end)
        return r_list_numpy_index, length

    def determine_real_long_path_by_curv(path, graphIndex_pointId_dict, curv_data_delete_thicknessZero):
        curv_of_path = list()
        for index in path:
            point = graphIndex_pointId_dict[index]
            curv_of_path.append(curv_data_delete_thicknessZero[point])
        return curv_of_path

    def determine_real_long_path(path, graphIndex_pointId_dict, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
        max_patchsize = 0
        for index in path:
            point = graphIndex_pointId_dict[index]
            if point_patchSize_dict_orig[point] > max_patchsize:
                max_patchsize = point_patchSize_dict_orig[point]
        if len(path) < 2 * max_patchsize:
            curv_of_path = Find_skelenton.determine_real_long_path_by_curv(path, graphIndex_pointId_dict, curv_data_delete_thicknessZero)
            curv_of_path_array = np.stack(curv_of_path)
            if np.sum(curv_of_path_array[0:6] > - 0.1) >= flat_threshold_for_convex_gyri:
                return False
            else:
                return True
        else:
            return True
    
    def build_missing_connection_recurrently(long_path_convex_points_list_sorted, final_point_lines_dict, point_connect_points_dict_gyri_parts, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
        missing_connection_list = list()
        for long_convex_marginal_point in long_path_convex_points_list_sorted:
            connection_dicts_list = Find_skelenton.find_connections_for_long_convex_marginal_point(long_convex_marginal_point,
                                                                                    final_point_lines_dict,
                                                                                    point_connect_points_dict_gyri_parts,
                                                                                    curv_data_delete_thicknessZero)
            if connection_dicts_list:
                from_marginal_to_skelenton_father_sons_dict = connection_dicts_list[0]
                from_skelenton_to_marginal_son_fathers_dict = connection_dicts_list[1]
                pointId_graphIndex_dict, graphIndex_pointId_dict = Find_skelenton.build_point_id_graphIndex_mapping(long_convex_marginal_point,
                                                                                                    from_skelenton_to_marginal_son_fathers_dict)
                graph = Find_skelenton.build_graph_for_convex_point(long_convex_marginal_point, pointId_graphIndex_dict,
                                                    from_marginal_to_skelenton_father_sons_dict,
                                                    curv_data_delete_thicknessZero)
                skelenton_point_list = [point for point in from_skelenton_to_marginal_son_fathers_dict.keys() if
                                        point in final_point_lines_dict.keys()]
                shortest_length = 100000000
                shortest_path = []
                for skelenton_point in skelenton_point_list:
                    skelenton_graph_id = pointId_graphIndex_dict[skelenton_point]
                    long_convex_marginal_point_id = pointId_graphIndex_dict[long_convex_marginal_point]
                    path_list, length = Find_skelenton.create_shortest_path(graph, long_convex_marginal_point_id, skelenton_graph_id)
                    if length < shortest_length:
                        shortest_path = path_list
                        shortest_length = length
                if shortest_path:
                    if Find_skelenton.determine_real_long_path(shortest_path, graphIndex_pointId_dict, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
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

    def build_main_skeleton_for_missing_island_gyri(longest_island_convex_point, final_point_lines_dict, point_connect_points_dict_gyri_parts, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri):
        print('build main skeleton for missing island \t' + time.asctime(time.localtime(time.time())))
        current_island_convex_point = longest_island_convex_point
        connection_dicts_list = Find_skelenton.find_connections_for_long_island_convex_marginal_point(current_island_convex_point,
                                                                                    point_connect_points_dict_gyri_parts,
                                                                                    curv_data_delete_thicknessZero)
        if connection_dicts_list:
            from_marginal_to_skelenton_father_sons_dict = connection_dicts_list[0]
            from_skelenton_to_marginal_son_fathers_dict = connection_dicts_list[1]
            pointId_graphIndex_dict, graphIndex_pointId_dict = Find_skelenton.build_point_id_graphIndex_mapping(current_island_convex_point,
                                                                                                from_skelenton_to_marginal_son_fathers_dict)
            graph = Find_skelenton.build_graph_for_convex_point(current_island_convex_point, pointId_graphIndex_dict,
                                                from_marginal_to_skelenton_father_sons_dict,
                                                curv_data_delete_thicknessZero)
            skelenton_point_list = connection_dicts_list[2]
            shortest_length = 100000000
            shortest_path = []
            for skelenton_point in skelenton_point_list:
                skelenton_graph_id = pointId_graphIndex_dict[skelenton_point]
                long_convex_marginal_point_id = pointId_graphIndex_dict[current_island_convex_point]
                path_list, length = Find_skelenton.create_shortest_path(graph, long_convex_marginal_point_id, skelenton_graph_id)
                if length < shortest_length:
                    shortest_path = path_list
                    shortest_length = length
            if Find_skelenton.determine_real_long_path(shortest_path, graphIndex_pointId_dict, point_patchSize_dict_orig,
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

    def create_connection_for_missing_gyri(orig_surf_polydata, orig_sphere_polydata, final_point_lines_dict, original_sulc_data, curv_data_delete_thicknessZero, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix):
        point_connect_points_dict_gyri_parts = get_connect_points_gyri_part(orig_sphere_polydata, original_sulc_data)
        point_neighbor_points_dict = get_connect_points(orig_sphere_polydata)
        point_patchSize_dict_orig = Find_skelenton.find_the_patchSize_of_gyri_point(original_sulc_data, point_neighbor_points_dict)
        draw_patchSize_colorful_file(orig_sphere_polydata, orig_surf_polydata, point_patchSize_dict_orig, output_prefix, 'orig')
        _, marginal_points_gyri, _ = find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, original_sulc_data)
        Find_skelenton.draw_3hinge_on_surf(orig_sphere_polydata, marginal_points_gyri, output_prefix + '_sphere_marginal.vtk')
        Find_skelenton.draw_3hinge_on_surf(orig_surf_polydata, marginal_points_gyri, output_prefix + '_surf_marginal.vtk')
        all_long_path_convex_points_list, convex_marginal_points_length_dict, island_convex_marginal_points_length_dict = Find_skelenton.find_convex_marginal_points_of_long_gyri(orig_sphere_polydata, final_point_lines_dict, point_neighbor_points_dict, point_connect_points_dict_gyri_parts, marginal_points_gyri, original_sulc_data, curv_data_delete_thicknessZero, point_patchSize_dict_orig, length_thres_of_long_gyri, nearest_skeleton_num, island_gyri_length_thres)
        Find_skelenton.draw_3hinge_on_surf(orig_sphere_polydata, all_long_path_convex_points_list, output_prefix + '_sphere_convex_marginal.vtk')
        Find_skelenton.draw_3hinge_on_surf(orig_surf_polydata, all_long_path_convex_points_list, output_prefix + '_surf_convex_marginal.vtk')
        long_path_convex_points_list = Find_skelenton.select_convex_endpoint_for_each_missing_gyri(all_long_path_convex_points_list, convex_marginal_points_length_dict, curv_data_delete_thicknessZero, point_connect_points_dict_gyri_parts, neighbor_missing_path_smallest_step)
        Find_skelenton.draw_3hinge_on_surf(orig_sphere_polydata, long_path_convex_points_list, output_prefix + '_sphere_convex_marginal_long.vtk')
        Find_skelenton.draw_3hinge_on_surf(orig_surf_polydata, long_path_convex_points_list, output_prefix + '_surf_convex_marginal_long.vtk')
        long_path_convex_points_list_sorted = Find_skelenton.sort_the_long_path_convex_point(long_path_convex_points_list, convex_marginal_points_length_dict)
        missing_connection_list, final_point_lines_dict = Find_skelenton.build_missing_connection_recurrently(long_path_convex_points_list_sorted, final_point_lines_dict, point_connect_points_dict_gyri_parts, point_patchSize_dict_orig, curv_data_delete_thicknessZero, flat_threshold_for_convex_gyri)
        Find_skelenton.write_skelenton_by_connectionPair(orig_sphere_polydata, missing_connection_list, output_prefix + '_sphere_missing_connection.vtk')
        Find_skelenton.write_skelenton_by_connectionPair(orig_surf_polydata, missing_connection_list, output_prefix + '_surf_missing_connection.vtk')

        long_path_island_convex_candicate_points_list = list(island_convex_marginal_points_length_dict.keys())
        while long_path_island_convex_candicate_points_list:
            long_path_island_convex_candicate_points_list_sorted = Find_skelenton.sort_the_long_path_convex_point(long_path_island_convex_candicate_points_list,
                                                                                                island_convex_marginal_points_length_dict)
            current_longest_island_convex_point = long_path_island_convex_candicate_points_list_sorted[0]
            final_point_lines_dict, connection_dicts_list = Find_skelenton.build_main_skeleton_for_missing_island_gyri(current_longest_island_convex_point,
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
            all_long_path_island_convex_points_list, convex_marginal_island_points_length_dict, _ = Find_skelenton.find_convex_marginal_points_of_long_gyri(
                orig_sphere_polydata, final_point_lines_dict, point_neighbor_points_dict,
                point_connect_points_dict_gyri_parts, island_convex_marginal_point_list, original_sulc_data,
                curv_data_delete_thicknessZero, point_patchSize_dict_orig, length_thres_of_long_gyri, nearest_skeleton_num,
                island_gyri_length_thres)
            long_path_island_convex_points_list = Find_skelenton.select_convex_endpoint_for_each_missing_gyri(all_long_path_island_convex_points_list,
                                                                                            convex_marginal_island_points_length_dict,
                                                                                            curv_data_delete_thicknessZero,
                                                                                            point_connect_points_dict_gyri_parts,
                                                                                            neighbor_missing_path_smallest_step)
            long_path_island_convex_points_list_sorted = Find_skelenton.sort_the_long_path_convex_point(long_path_island_convex_points_list,
                                                                                        convex_marginal_island_points_length_dict)
            _, final_point_lines_dict = Find_skelenton.build_missing_connection_recurrently(long_path_island_convex_points_list_sorted,
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

    def find_skelenton(orig_sphere_polydata, orig_surf_polydata, point_patchSize_dict_updated, curv_data_delete_thicknessZero, original_sulc_data, thin_sulc_data, point_neighbor_points_dict, point_connect_points_dict_thin_gyri_parts, connected_lines_list, father_dict, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix, sphere):
        print('================= build skeleton:\t' + time.asctime(time.localtime(time.time())) + '=======================')
        marginal_points, marginal_points_gyri, marginal_points_sulc = find_marginal_point(list(point_neighbor_points_dict.keys()), point_neighbor_points_dict, thin_sulc_data)

        print('begin trimming  \t' + time.asctime(time.localtime(time.time())))
        tree_point_lines_dict = Find_skelenton.create_tree_connection_dict(connected_lines_list)
        inner_gyri_points = [gyri_point for gyri_point in tree_point_lines_dict if gyri_point not in marginal_points_gyri]
        initial_skelenton_point_lines_dict, deleted_round1_points = Find_skelenton.trim_allRound1_long(tree_point_lines_dict, point_patchSize_dict_updated, point_connect_points_dict_thin_gyri_parts, father_dict, inner_gyri_points)
        initial_skelenton_connection_list = Find_skelenton.create_connectingPair(initial_skelenton_point_lines_dict)
        print('draw initial skelenton  \t' + time.asctime(time.localtime(time.time())))

        sphere_initial_skelenton_path = os.path.join(output_prefix, "%s_sphere_initial_skelenton.vtk"%(sphere))
        surf_initial_skelenton_path =  os.path.join(output_prefix, "%s_surf_initial_skelenton.vtk"%(sphere))
        Find_skelenton.write_skelenton_by_connectionPair(orig_sphere_polydata, initial_skelenton_connection_list, sphere_initial_skelenton_path)
        Find_skelenton.write_skelenton_by_connectionPair(orig_surf_polydata, initial_skelenton_connection_list, surf_initial_skelenton_path)

        print('connecting breaks in skelenton  \t' + time.asctime(time.localtime(time.time())))
        final_point_lines_dict = Find_skelenton.connect_break_in_circle(initial_skelenton_point_lines_dict, point_connect_points_dict_thin_gyri_parts, deleted_round1_points, father_dict)
        final_connection_list = Find_skelenton.create_connectingPair(final_point_lines_dict)

        sphere_skelenton_connect_break_allpoints_path = os.path.join(output_prefix, "%s_sphere_skelenton_connect_break_allpoints.vtk"%(sphere))
        surf_skelenton_connect_break_allpoints_path = os.path.join(output_prefix, "%s_surf_skelenton_connect_break_allpoints.vtk"%(sphere))
        Find_skelenton.write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, final_connection_list, sphere_skelenton_connect_break_allpoints_path)
        Find_skelenton.write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata, final_connection_list, surf_skelenton_connect_break_allpoints_path)

        final_point_lines_dict = Find_skelenton.clear_empty_point_in_dict(final_point_lines_dict)

        print('find missing gyri  \t' + time.asctime(time.localtime(time.time())))
        final_point_lines_dict = Find_skelenton.create_connection_for_missing_gyri(orig_surf_polydata, orig_sphere_polydata, final_point_lines_dict, original_sulc_data, curv_data_delete_thicknessZero, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix)
        all_connection_list = Find_skelenton.create_connectingPair(final_point_lines_dict)

        print('draw final skelenton  \t' + time.asctime(time.localtime(time.time())))
        sphere_skelenton_allpoints_final_path = os.path.join(output_prefix, "%s_sphere_skelenton_allpoints_final.vtk"%(sphere))
        surf_skeleton_allpoints_final_path = os.path.join(output_prefix, "%s_surf_skelenton_allpoints_final.vtk"%(sphere))        
        Find_skelenton.write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, all_connection_list, sphere_skelenton_allpoints_final_path)
        Find_skelenton.write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata, all_connection_list, surf_skeleton_allpoints_final_path)

        all_point_lines_dict = Find_skelenton.create_tree_connection_dict(all_connection_list)

        print('find and draw 3hinge' + time.asctime(time.localtime(time.time())))
        hinge3_list = Find_skelenton.create_3hinge(all_point_lines_dict)
        hinge3_txt = open(output_prefix + '_3hinge_ids.txt', "w")
        for hinge in hinge3_list:
            hinge3_txt.write(str(hinge) + "\n")
        hinge3_txt.close()

        sphere_3hinge_vertex_path = os.path.join(output_prefix, "%s_sphere_3hinge_vertex.vtk"%(sphere))
        surf_3hinge_vertex_path = os.path.join(output_prefix, "%s_surf_3hinge_vertex.vtk"%(sphere))
        Find_skelenton.draw_3hinge_on_surf(orig_sphere_polydata, hinge3_list, sphere_3hinge_vertex_path)
        Find_skelenton.draw_3hinge_on_surf(orig_surf_polydata, hinge3_list, surf_3hinge_vertex_path)

    def read_connection_of_skelenton_file(skelenten_polydata):
        connected_lines_list = list()
        line_num = skelenten_polydata.GetNumberOfCells()
        for line in range(line_num):
            pointID1 = skelenten_polydata.GetCell(line).GetPointIds().GetId(0)
            pointID2 = skelenten_polydata.GetCell(line).GetPointIds().GetId(1)
            connected_lines_list.append([pointID1, pointID2])
        return connected_lines_list

    def find_skelenton_missing(orig_sphere_polydata, orig_surf_polydata, skeleton_polydata, curv_data_delete_thicknessZero, original_sulc_data, length_thres_of_long_gyri, neighbor_missing_path_smallest_step, flat_threshold_for_convex_gyri, nearest_skeleton_num, island_gyri_length_thres, output_prefix,sphere):
        print('================= build skeleton:\t' + time.asctime(time.localtime(time.time())) + '=======================')
        final_connection_list = Find_skelenton.read_connection_of_skelenton_file(skeleton_polydata)
        final_point_lines_dict = Find_skelenton.create_tree_connection_dict(final_connection_list)
        print('find missing gyri  \t' + time.asctime(time.localtime(time.time())))
        final_point_lines_dict = Find_skelenton.create_connection_for_missing_gyri(orig_surf_polydata, orig_sphere_polydata,
                                                                    final_point_lines_dict, original_sulc_data,
                                                                    curv_data_delete_thicknessZero,
                                                                    length_thres_of_long_gyri,
                                                                    neighbor_missing_path_smallest_step,
                                                                    flat_threshold_for_convex_gyri, nearest_skeleton_num,
                                                                    island_gyri_length_thres, output_prefix)
        all_connection_list = Find_skelenton.create_connectingPair(final_point_lines_dict)

        sphere_skelenton_allpoints_final_path = os.path.join(output_prefix, "sphere_skelenton_allpoints_final.vtk")
        surf_skeleton_allpoints_final_path = os.path.join(output_prefix, "surf_skelenton_allpoints_final.vtk")
        Find_skelenton.write_allPoints_and_skelenton_by_connectionPair(orig_sphere_polydata, all_connection_list, sphere_skelenton_allpoints_final_path)
        Find_skelenton.write_allPoints_and_skelenton_by_connectionPair(orig_surf_polydata, all_connection_list, surf_skeleton_allpoints_final_path)

        all_point_lines_dict = Find_skelenton.create_tree_connection_dict(all_connection_list)

        print('find and draw 3hinge' + time.asctime(time.localtime(time.time())))
        hinge3_list = Find_skelenton.create_3hinge(all_point_lines_dict)
        hinge3_txt = open(output_prefix + '_3hinge_ids.txt', "w")
        for hinge in hinge3_list:
            hinge3_txt.write(str(hinge) + "\n")
        hinge3_txt.close()

        sphere_3hinge_vertex_path = os.path.join(output_prefix, "%s_sphere_3hinge_vertex.vtk"%(sphere))
        surf_3hinge_vertex_path = os.path.join(output_prefix, "%s_surf_3hinge_vertex.vtk"%(sphere))
        Find_skelenton.draw_3hinge_on_surf(orig_sphere_polydata, hinge3_list, sphere_3hinge_vertex_path)
        Find_skelenton.draw_3hinge_on_surf(orig_surf_polydata, hinge3_list, surf_3hinge_vertex_path)
        
if __name__=="__main__":
    print("True")