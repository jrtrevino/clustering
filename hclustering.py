import json
import math
import re
import sys
import utility

import numpy as np
import pandas as pd

from utility import csv_to_df


def find_closest_index(df):
    # need to dendrogram it further with tuples
    min_index = None
    min_num = math.inf
    start = 1
    for index, row in df.iterrows():
        current_column = 0
        for col, val in row.iteritems():
            if current_column >= start and val != 0:
                current_value = val
                if current_value < min_num:
                    min_num = val
                    if type(eval(str(index))) == int:
                        index = int(index)
                    else:
                        index = eval(str(index))

                    if type(eval(col[0])) == int:
                        col_val = int(col[0])
                    else:
                        col_val = eval(col[0])
                    min_index = (index, col_val)
            current_column += 1
        start += 1
    return min_index, min_num


def euclidean(x, y):
    return np.sqrt(np.sum((x - y) * (x - y)))


def manhattan(x, y):
    return np.sum(np.abs(x - y))


# use a distance metric to find two points
def get_distance(dp1, dp2, metric):
    dp1 = np.array(dp1)
    dp2 = np.array(dp2)
    if metric == 0:
        return euclidean(dp1, dp2)
    return manhattan(dp1, dp2)


# create distance matrix from a dataframe
def gen_distance_df(dataframe):
    # our dataframe matrix
    matrix_df = pd.DataFrame()
    for point in range(len(dataframe)):
        distances = [-1] * len(dataframe)
        point_names = [None] * len(dataframe)
        for point2 in range(len(dataframe)):
            if point == point2:
                distance = 0
            else:
                # using Euclidean  distance as default distance metric
                distance = get_distance(dataframe.iloc[[point]], dataframe.iloc[[point2]], 0)
            distances[point2] = distance
            point_names[point2] = '{}'.format(dataframe.iloc[point2].name)
        temp_df = pd.DataFrame([distances], columns=[point_names], index=[point_names[point]])
        matrix_df = matrix_df.append(temp_df)
    return matrix_df


# merges two clusters into one
def merge_data(point1, point2, index):
    merged = []
    for i in range(len(point1.values[0])):
        merged.append((point1.values[0][i] + point2.values[0][i]) / 2)
    return merged


cluster_lookup = {}


def merge_matrix_df(data_df, index_arg):
    # turn indices into dataframes
    data_df.index = data_df.index.map(str)
    first_point = str(index_arg[0])
    second_point = str(index_arg[1])
    # print(str(first_point) + ', ' + str(second_point))
    dp1 = data_df.loc[[first_point]]
    dp2 = data_df.loc[[second_point]]
    merged_dp = merge_data(dp1, dp2, index_arg)
    temp_df = pd.DataFrame([merged_dp], columns=data_df.columns, index=[index_arg])
    data_df = data_df.append(temp_df)
    # drop old labels
    data_df = data_df.drop([first_point, second_point])
    return data_df


def get_qualifying_clusters(threshold):
    clusters = [key for key, value in cluster_lookup.items() if value >= threshold]
    return clusters


def split(distance_dict, threshold, cluster_arr, master):
    dict_list = sorted(distance_dict.items(),
                       key=lambda entry: entry[1], reverse=True
                       )
    first_item = dict_list[0]  # tuple
    distance_dict.pop(first_item[0], None)
    if first_item[1] <= threshold:
        cluster_arr.append(first_item[0])
    else:
        # split the key
        cluster_one = first_item[0][0]
        cluster_two = first_item[0][1]
        if type(cluster_one) is int:
            cluster_arr.append(cluster_one)
        else:
            if master[cluster_one] <= threshold:
                cluster_arr.append(cluster_one)
            else:
                split(distance_dict, threshold, cluster_arr, master)
        if type(cluster_two) is int:
            cluster_arr.append(cluster_two)
        else:
            if master[cluster_two] <= threshold:
                cluster_arr.append(cluster_two)
            else:
                split(distance_dict, threshold, cluster_arr, master)


def eval_clusters(dendrogram, df, restrictions, threshold=None):
    if threshold is not None:
        cluster_arr = []
        split(dendrogram.copy(), threshold, cluster_arr, dendrogram.copy())
        sse = 0
        cluster_length = len(cluster_arr)
        for cluster in cluster_arr:
            temp_df = pd.DataFrame()
            if type(cluster) is int:
                temp_df = temp_df.append(df.loc[[str(cluster)]])
            else:
                temp = ("{}".format(cluster).replace("(", "").replace(")", ""))
                temp = temp.split(",")
                temp = [int(x.strip()) for x in temp]
                for index in temp:
                    temp_df = temp_df.append(df.loc[[str(index)]])
                # conver temp to dataframe
                # calculate dataframe mean
                # calculate dataframe SSE
            mean = utility.get_df_mean(temp_df, restrictions, pr=False)
            print(f'cluster sse {utility.calculate_sse([temp_df], [mean])}')
            sse += utility.calculate_sse([temp_df], [mean])
            print(temp_df, "\n")
        print(f"cluster sse: {sse} , cluster length: {cluster_length} threshold: {threshold}")
        print("\n")
    else:
        for value in dendrogram.copy().values():
            cluster_arr = []
            split(dendrogram.copy(), value, cluster_arr, dendrogram.copy())
            sse = 0
            cluster_length = len(cluster_arr)
            for cluster in cluster_arr:
                temp_df = pd.DataFrame()
                if type(cluster) is int:
                    temp_df = temp_df.append(df.loc[[str(cluster)]])
                else:
                    temp = ("{}".format(cluster).replace("(", "").replace(")", ""))
                    temp = temp.split(",")
                    temp = [int(x.strip()) for x in temp]
                    for index in temp:
                        temp_df = temp_df.append(df.loc[[str(index)]])
                    # conver temp to dataframe
                    # calculate dataframe mean
                    # calculate dataframe SSE
                mean = utility.get_df_mean(temp_df, restrictions, pr=False)
                sse += utility.calculate_sse([temp_df], [mean])
                # print(temp_df, "\n")
            print(f"cluster sse: {sse} , cluster length: {cluster_length} threshold: {value}")
            print("\n")


"""
calculate mean and sse
find smallest sse without large # of clusters 
"""


def create_json(df, tuple_cluster, root=False):
    data = {
        "type": "node",
        "nodes": []
    }
    left = tuple_cluster[0]
    right = tuple_cluster[1]
    if root:
        data['height'] = cluster_lookup[tuple_cluster]
        data['type'] = 'root'
        if type(left) == int:
            leaf_left = {
                "type": "leaf",
                "height": 0,
                "data": left
            }
            data['nodes'].append(leaf_left)
        else:
            data['nodes'].append(create_json(df, left))
        if type(right) == int:
            leaf_right = {
                "type": "leaf",
                "height": 0,
                "data": right
            }
            data['nodes'].append(leaf_right)
        else:
            data['nodes'].append(create_json(df, right))
    elif type(left) == int and type(right) == int:
        leaf_left = {
            "type": "leaf",
            "height": 0,
            "data": left
        }
        leaf_right = {
            "type": "leaf",
            "height": 0,
            "data": right
        }
        return [leaf_left, leaf_right]
    else:
        left_leaf = right_leaf = None
        if type(left) == int:
            left_leaf = {
                "type": "leaf",
                "height": 0,
                "data": left
                # "data": df.loc[str(left)]
            }
            data["nodes"].append(left_leaf)
        else:
            data["nodes"].append(create_json(df, left))
            data["height"] = cluster_lookup[left]
        if type(right) == int:
            right_leaf = {
                "type": "leaf",
                "height": 0,
                "data": right
                # "data": df.loc[str(right)]
            }
            data["nodes"].append(right_leaf)
            data["height"] = cluster_lookup[right]
        else:
            data["nodes"].append(create_json(df, right))
    return data


def print_clusters(clusters, df):
    for index in range(len(clusters)):
        print(f"Cluster {index}: ")
        datapoints = [int(data) for data in re.findall(r'\b\d+\b', str(clusters[index]))]
        print(f"{len(datapoints)} points:")
        [print(f"Point: {point} {df.loc[[str(point)]]}") for point in datapoints]


def write_json_file(csv_file, dendrogram):
    with open(str(csv_file + '.json'), 'w') as conversion:
        conversion.write(json.dumps(dendrogram))


# main function
# csv file -> string tuple representing cluster
def hcluster(csv_file, threshold=None):
    df, restrictions = csv_to_df(csv_file)
    dropped = []
    for column, index in zip(df.columns, restrictions):
        if index == 0:
            dropped.append(column)
    df = df.drop(dropped, axis=1)
    df.index = df.index.map(str)
    changing_df = df.copy()
    while len(changing_df) > 1:
        distance_matrix = gen_distance_df(changing_df)
        index, min_num = find_closest_index(distance_matrix)
        cluster_lookup[index] = min_num
        merged_df = merge_matrix_df(changing_df, index)
        changing_df = merged_df
    root_cluster = changing_df.index.tolist()
    dendro_json = create_json(df, root_cluster[0], True)
    if threshold:
        eval_clusters(cluster_lookup, df, restrictions, float(threshold))
    else:
        eval_clusters(cluster_lookup, df, restrictions, None)
    write_json_file(csv_file, dendro_json)
    return dendro_json


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 1:
        dendro = hcluster(args[0], args[1])
    else:
        dendro = hcluster(args[0])
