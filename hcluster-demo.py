import pandas as pd
import numpy as np
from utility import csv_to_df


def find_closest_index(df):
    indices = (np.random.randint(0, high=len(df) - 1, size=None, dtype=int), 
    np.random.randint(0, high=len(df) - 1, size=None, dtype=int))
    pt1 = df.index[indices[0]]
    pt2 = df.index[indices[1]]
    return (pt1, pt2)

# use a distance metric to find two points
def gen_random_distance(dp1, dp2):
    return np.random.randint(1, high=20, size=None, dtype=int)


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
                distance = gen_random_distance(dataframe.iloc[[point]], dataframe.iloc[[point2]])
            distances[point2] = distance
            point_names[point2] = '{}'.format(dataframe.iloc[point2].name)
        temp_df = pd.DataFrame([distances], columns=[point_names], index=[point_names[point]])
        matrix_df = matrix_df.append(temp_df)
    return matrix_df


# merges two clusters into one 
def merge_data(point1, point2, index):
    merged = []
    for i in range(len(point1.values[0])):
        merged.append((point1.values[0][i] + point2.values[0][i])/2)
    return merged
   

def merge_matrix_df(data_df, index_arg):
    # turn indices into dataframes 
    dp1 = data_df.loc[[index_arg[0]]]
    dp2 = data_df.loc[[index_arg[1]]]
    merged_dp = merge_data(dp1, dp2, index_arg)
    temp_df = pd.DataFrame([merged_dp], columns=data_df.columns, index=[index_arg])
    data_df = data_df.append(temp_df)
    # drop old labels
    data_df = data_df.drop([index_arg[0], index_arg[1]])
    return data_df


## SETUP
df, restrictions = csv_to_df('./data/4clusters.csv')
# initial distance matrix (step 1)
distance_matrix = gen_distance_df(df)
# find closest index (step 2)
index = find_closest_index(distance_matrix)
# merge dataframe based on closest index (step 3)
merged_df = merge_matrix_df(df, index)
print(merged_df)
# Repeat step 1-3 with merged_df
new_distance_matrix = gen_distance_df(merged_df)
index = find_closest_index(new_distance_matrix)
new_merged_df = merge_matrix_df(merged_df, index)
print(new_merged_df)
# Repeat step 1-3 with merged_df
new_distance_matrix = gen_distance_df(new_merged_df)
index = find_closest_index(new_distance_matrix)
new_merged_df = merge_matrix_df(new_merged_df, index)
print(new_merged_df)
# Repeat step 1-3 with merged_df
new_distance_matrix = gen_distance_df(new_merged_df)
index = find_closest_index(new_distance_matrix)
new_merged_df = merge_matrix_df(new_merged_df, index)
print(new_merged_df)
# Repeat step 1-3 with merged_df
new_distance_matrix = gen_distance_df(new_merged_df)
index = find_closest_index(new_distance_matrix)
new_merged_df = merge_matrix_df(new_merged_df, index)
print(new_merged_df)
# Repeat step 1-3 with merged_df
new_distance_matrix = gen_distance_df(new_merged_df)
index = find_closest_index(new_distance_matrix)
new_merged_df = merge_matrix_df(new_merged_df, index)
print(new_merged_df)
# Repeat step 1-3 with merged_df
new_distance_matrix = gen_distance_df(new_merged_df)
index = find_closest_index(new_distance_matrix)
new_merged_df = merge_matrix_df(new_merged_df, index)
# Repeat step 1-3 with merged_df
new_distance_matrix = gen_distance_df(new_merged_df)
index = find_closest_index(new_distance_matrix)
new_merged_df = merge_matrix_df(new_merged_df, index)
# Repeat step 1-3 with merged_df
new_distance_matrix = gen_distance_df(new_merged_df)
index = find_closest_index(new_distance_matrix)
new_merged_df = merge_matrix_df(new_merged_df, index)
# Repeat step 1-3 with merged_df
new_distance_matrix = gen_distance_df(new_merged_df)
index = find_closest_index(new_distance_matrix)
new_merged_df = merge_matrix_df(new_merged_df, index)


