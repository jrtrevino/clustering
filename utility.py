import pandas as pd
import math
#import matplotlib.pyplot as plt


def csv_to_df(path_to_csv):
    counter = 0
    with open(path_to_csv, "r") as file:
        restrictions = list(map(int, file.readline().rstrip().split(',')))
    headers = get_headers(path_to_csv)
    df = pd.read_csv(path_to_csv, skiprows=0, names=headers)
    for column in df:
        if restrictions[counter] == 1:
            df[column] = pd.to_numeric(df[column], errors='coerce')
            max_value = df[column].max()
            min_value = df[column].min()
            df[column] = (df[column] - min_value) / (max_value - min_value)
        counter += 1
    return df, restrictions


def get_df_mean(df, restrictions, pr=False):
    if pr:
        print("df_mean {}".format(df))
    mean_vector = [0] * len(df.columns)
    for i in range(len(df.columns)):
        if restrictions[i] == 0:
            mean_vector[i] = 'NaN'
            continue
        mean_vector[i] = df[df.columns[i]].mean()
    return mean_vector


def calculate_sse(clusters, cluster_means):
    sse_arr = [0] * len(clusters)
    counter = 0
    for cluster in clusters:
        sse = 0
        for index, row in cluster.iterrows():
            values = row.tolist()
            # distance from row[column] from mean[column]
            sse += sum([(values[column_index] - cluster_means[counter][column_index])**2
                       for column_index in range(len(cluster_means[counter])) if cluster_means[counter][column_index] != 'NaN'])
        sse_arr[counter] = sse/len(cluster)
        counter += 1
    return sum(sse_arr)


def get_max_min_distance(df, point):
    max_d = min_d = None
    avg_d = 0
    for index, row in df.iterrows():
        row_d = get_euclid_distance(df.loc[[index]], point)
        if max_d is None or min_d is None:
            max_d = min_d = row_d
        elif row_d > max_d:
            max_d = row_d
        else:
            min_d = row_d
        avg_d += row_d
    return [max_d, min_d, avg_d/len(df)]


def get_euclid_distance(cluster_point, mean_vector):
    distance = counter = 0
    for column in cluster_point:
        col_val = cluster_point[column].values[0]
        if type(col_val) is str:
            continue
        col_mean = mean_vector[counter]
        if col_mean != 'NaN' and type(col_mean) is not str:
            distance += (col_val - col_mean)**2
        counter += 1
    return math.sqrt(distance)


def draw_clusters(clusters):
    colors = ['red', 'blue', 'purple', 'yellow', 'green', 'orange']
    counter = 0
    ax = clusters[0]
    ax1 = ax.plot(
        kind='scatter', x=clusters[0].columns[0], y=clusters[0].columns[1], color=colors[0])
    for i in range(1, len(clusters)):
        clusters[i].plot(kind='scatter', x=clusters[i].columns[0],
                         y=clusters[i].columns[1], color=colors[i],  ax=ax1)


def get_headers(path_to_csv):
    if 'mammal' in path_to_csv:
        return ['Animal', 'Water', 'Protein', 'Fat', 'Lactose', 'Ash']
    elif 'Set01' in path_to_csv:
        return ['VE_TOTAL', 'PERSONS', 'FATALS']
    elif 'Set03' in path_to_csv:
        return ['VE_TOTAL', 'PEDS', 'NO_LANES',  'FATALS', 'DRUNK_DR']
    elif 'clusters' in path_to_csv or 'Clusters' in path_to_csv:
        return ['Col1', 'Col2']
    return ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
