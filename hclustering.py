import math
import sys

import numpy as np
import pandas as pd

from utility import csv_to_df


def find_closest_index(df):
    # need to test it further with tuples
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
    return min_index


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


def create_json(root_cluster):
    print(type(root_cluster))
    tree = {
        "type": "root",
        "height": 1,
        "nodes": []
    }
    value = eval(root_cluster)
    return 0


# main function
# csv file -> string tuple representing cluster
def hcluster(csv_file):
    df, restrictions = csv_to_df(csv_file)
    df.index = df.index.map(str)
    changing_df = df.copy()
    while len(changing_df) > 1:
        distance_matrix = gen_distance_df(changing_df)
        index = find_closest_index(distance_matrix)
        merged_df = merge_matrix_df(changing_df, index)
        changing_df = merged_df
    root_cluster = changing_df.index.tolist()
    dendro_json = create_json(root_cluster[0])
    return dendro_json


if __name__ == "__main__":
    args = sys.argv[1:]
    hcluster(args[0])
