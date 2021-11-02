import math
import pandas as pd
import sys
from utility import csv_to_df


def kmeans(df, restrictions, centroids):
    # we only need to do the next two lines if our centroids are actual points.
    # if they are averages, we can skip to reassignment
    clusters = [df.loc[[df.index[centroids[x]]]]
                for x in range(len(centroids))]
    cluster_means = [get_df_mean(clusters[x], restrictions)
                     for x in range(len(centroids))]
    cluster_assignment = [0] * len(centroids)
    # previous_assignment = [0] * len(centroids)
    sse_threshold = 0.10
    # assignment_threshold = 0.1
    prev_sse = [1] * len(centroids)
    # print("Initial cluster means: {}".format(cluster_means))
    while (True):
        for index, row in df.iterrows():
            distances = [] * len(clusters)
            for i in range(len(clusters.copy())):
                distances.append((i, get_euclid_distance(
                    df.loc[[index]], cluster_means[i])))
            cluster_selection = sorted(
                distances, key=lambda tuple: tuple[1])[0]
            # check if data is already in the same cluster
            in_cluster = [index in clusters[x].index.values for x in range(
                len(clusters))]
            if True in in_cluster:
                cluster_index = in_cluster.index(True)
                clusters[cluster_index] = clusters[cluster_index].drop(index)
            clusters[cluster_selection[0]] = clusters[cluster_selection[0]].append(
                df.loc[[index]])
            cluster_assignment[cluster_selection[0]] += 1
        cluster_means = [get_df_mean(clusters[x], restrictions)
                         for x in range(len(centroids))]
        sse_arr = calculate_sse(clusters, cluster_means)
        if False not in [(sse_arr[sse] == prev_sse[sse] or abs(sse_arr[sse] - prev_sse[sse]) / prev_sse[sse] <= sse_threshold) for sse in range(len(sse_arr))]:
            # print("sse ratio: {}".format(sse/prev_sse))
            # print("sse bound")
            break
        prev_sse = sse_arr
        # print("Assignment numbers: {}".format(cluster_assignment))
        # print("SSE: {}".format(sse))
        # check stopping criterion
        # previous_assignment = cluster_assignment.copy()
        cluster_assignment = [0] * len(centroids)
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
    return sse_arr

    # Randomly samples k datapoints from df to use as initial centroids
    # for k-means.
    # INPUT:
    #    df -> a pandas dataframe
    #    k -> an integer representing number of centroids to pick
    # OUTUT:
    #    centroids -> a list containing the row-indices of the datapoints
    #    chosen for our initial centroids.


def get_k_centroids_random(df, k):
    centroids = df.sample(n=k, random_state=1)
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


def get_df_mean(df, restrictions):
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


def draw_clusters(clusters):
    for cluster in clusters:
        columns = cluster.columns
        cluster.plot(x=columns[0], y=columns[1])


# A wrapper for our kmeans function that we use if the program is started from main


def wrapper(dataset, k):
    df, restrictions = csv_to_df(dataset)
    # get initial centroids
    centroids = get_k_centroids_random(df, k)
    clusters = kmeans(df, restrictions, centroids)
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
