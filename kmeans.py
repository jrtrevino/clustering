import math
import numpy as np
import pandas as pd
import sys
from utility import csv_to_df
import matplotlib.pyplot as plt


def kmeans(df, restrictions, centroids, iterations):
    clusters = [df.loc[[x]] for x in centroids]
    cluster_centers = [get_df_mean(x, restrictions) for x in clusters]
    sse_threshold = 0.10
    prev_sse = None
    for i in range(iterations):
        for index, row in df.iterrows():
            if index in centroids:
                continue  # prevent adding the centroid point to the cluster again
            distances = sorted(
                [(get_euclid_distance(df.loc[[index]], cluster_centers[y]), y) for y in range(len(cluster_centers))])
            for cluster_num in range(len(clusters)):
                if index in clusters[cluster_num].index:
                    clusters[cluster_num] = clusters[cluster_num].drop(index)
            clusters[distances[0][1]] = clusters[distances[0]
                                                 [1]].append(df.loc[[index]])
        cluster_centers = [get_df_mean(x, restrictions) for x in clusters]
        sse = calculate_sse(clusters, cluster_centers)
        if prev_sse == None:
            prev_sse = sse
        else:
            if abs((prev_sse - sse)/prev_sse) <= sse_threshold:
                break
            prev_sse = sse
    return clusters


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


def get_k_centroids_max_distance(df, restrictions, k):
    centroids = pd.DataFrame()
    df_sample = df.sample(frac=0.2, random_state=21)
    df_center = get_df_mean(df_sample, restrictions)
    # get clusters
    for i in range(k):
        max_distance = max_index = None
        if i == 0 or i == 1:
            for index, row in df_sample.iterrows():
                point = df_center if i == 0 else get_df_mean(
                    centroids.iloc[[0]], restrictions)
                distance = get_euclid_distance(df_sample.loc[[index]], point)
                if max_distance is None or distance > max_distance:
                    max_distance = distance
                    max_index = index
            centroids = centroids.append(df_sample.loc[[max_index]])
        else:
            for index, row in df_sample.iterrows():
                points = [get_df_mean(centroids.iloc[[x]], restrictions)
                          for x in range(len(centroids))]
                distance = sum([get_euclid_distance(
                    df_sample.loc[[index]], point) for point in points])
                if max_distance is None or distance > max_distance:
                    max_distance = distance
                    max_index = index
            centroids = centroids.append(df_sample.loc[[max_index]])

    return centroids.index.tolist()


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


# Randomly samples k datapoints from df to use as initial centroids
# for k-means.
# INPUT:
#    df -> a pandas dataframe
#    k -> an integer representing number of centroids to pick
# OUTPUT:
#    centroids -> a list containing the row-indices of the datapoints
#    chosen for our initial centroids.


def get_k_centroids_random(df, k):
    centroids = df.sample(n=k)
    return centroids.index.tolist()

# Used for recalculating cluster centers.
# Calculates the mean vector of a dataframe, df.
# Columns of a dataframe not used for mean calculation are
# labeled as a NaN string.
# INPUT:
#     df -> a pandas dataframe (should be a cluster)
# OUTPUT:
#     mean_vector ->an array of means where each entry i is the mean
#     for column i.


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

# Calculates the SSD (sum of squared distance) between a datapoint
# and a cluster.
# INPUT: cluster_point -> a row of a dataframe to compute distance on
#        mean_vector -> a vector (list) representing the mean values
#        of a cluster.
# OUTPUT: distance -> a float value representing the distance from cluster_point
# to a cluster.


def get_euclid_distance(cluster_point, mean_vector):
    distance = counter = 0
    for column in cluster_point:
        col_val = cluster_point[column].values[0]
        col_mean = mean_vector[counter]
        if col_mean != 'NaN':
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
        if len(clusters) == 1:
            break
        clusters[i].plot(kind='scatter', x=clusters[i].columns[0],
                         y=clusters[i].columns[1], color=colors[i],  ax=ax1)

# A wrapper for our kmeans function that we use if the program is started from main


def wrapper(dataset, k):
    df, restrictions = csv_to_df(dataset)
    merged = pd.DataFrame()
    # get initial centroids
    centroids = get_k_centroids_max_distance(df, restrictions, k)
    clusters = kmeans(df, restrictions, centroids, 20)
    counter = 0
    for cluster in clusters:
        mean = get_df_mean(cluster, restrictions)
        distances = get_max_min_distance(cluster, mean)
        print("Cluster {}:".format(counter))
        print("Center: {}".format(mean))
        print("Max distance to center: {}".format(distances[0]))
        print("Min distance to center: {}".format(distances[1]))
        print("Avg distance to center: {}".format(distances[2]))
        print("Cluster SSE: {}".format(calculate_sse([cluster], [mean])))
        print("{} Points:".format(len(cluster)))
        for index, row in cluster.iterrows():
            print(index, row.values)
        print("\n")
        counter += 1
    draw_clusters(clusters)


if __name__ == "__main__":
    args = sys.argv[1:]
    # check arguments
    if len(args) != 2:
        print("Usage: kmeans.py <dataset> <k>")
        sys.exit(-1)
    elif '.csv' not in args[0] or not args[1].isdigit():
        print("First argument must be a csv file and second argument must be an integer.")
        print("Usage: kmeans.py <dataset> <k>")
        sys.exit(-1)
    # begin program
    wrapper(args[0], int(args[1]))
