import utility
import numpy as np
import pandas as pd
import sys


def dbscan(df, min_points, epsilon, restrictions):
    core_points = []
    outliers = []
    clusters = []
    outliers_df = pd.DataFrame()
    visited_points = set()  # df indices
    for index, row in df.iterrows():
        if index in visited_points:
            continue
        visited_points.add(index)
        neighbor_indices = nearest_points(
            df, df.loc[[index]], epsilon)
        if len(neighbor_indices) >= min_points:
            core_points.append(({index}, set()))
            merger = set(neighbor_indices)
            while merger:
                neighbor = merger.pop()
                neighbors_neighbor = nearest_points(
                    df, df.loc[[neighbor]], epsilon)
                if neighbor not in visited_points:
                    visited_points.add(neighbor)
                    if len(neighbors_neighbor) >= min_points:
                        merger |= set(neighbors_neighbor)
                if not any([neighbor == core[0] or neighbor in core[1] for core in core_points]):
                    # check if core
                    core_points[-1][0].add(neighbor) if len(
                        neighbors_neighbor) >= min_points else core_points[-1][1].add(neighbor)
        else:
            outliers.append(index)
    # format for drawing
    for core in core_points:
        temp_df = pd.DataFrame()
        for core_point in core[0]:
            temp_df = temp_df.append(df.loc[[core_point]])
        for reachable_point in core[1]:
            temp_df = temp_df.append(df.loc[[reachable_point]])
        clusters.append(temp_df)
    for outlier in outliers:
        outliers_df = outliers_df.append(df.loc[[outlier]])
    return clusters, outliers_df


# filters dataframe for all points within a provided distance of
# a provided point.
# Returns a list of indices of these points in reference to the provied
# dataframe.
def nearest_points(df, point, distance):
    nearest_points = df.apply(
        lambda data: utility.get_euclid_distance(point, data), axis=1
    ).where(lambda x: x <= distance
            ).where(lambda x: x != 0
                    ).dropna()
    return nearest_points.index.tolist()


def wrapper(dataset, min_points, epsilon):
    df, restrictions = utility.csv_to_df(dataset)
    # print(df, restrictions)
    #nearest_indices = nearest_points(df, df.iloc[[8]], epsilon)
    clusters, out = dbscan(df, min_points, epsilon, restrictions)
    counter = 0
    for cluster in clusters:
        mean = utility.get_df_mean(cluster, restrictions)
        distances = utility.get_max_min_distance(cluster, mean)
        print("Cluster {}:".format(counter))
        print("Center: {}".format(mean))
        print("Max distance to center: {}".format(distances[0]))
        print("Min distance to center: {}".format(distances[1]))
        print("Avg distance to center: {}".format(distances[2]))
        print("Cluster SSE: {}".format(
            utility.calculate_sse([cluster], [mean])))
        print("{} Points:".format(len(cluster)))
        for index, row in cluster.iterrows():
            print(index, row.values)
        print("\n")
        counter += 1
    print("Outliers:\n {}".format(out))
    if len(clusters) > 0:
        utility.draw_clusters(clusters)


if __name__ == "__main__":
    args = sys.argv[1:]
    # check arguments
    if len(args) != 3:
        print("Usage: dbscan.py <dataset> <min_points> <epsilon>")
        sys.exit(-1)
    elif '.csv' not in args[0] or not args[1].isdigit() or not args[2].isdigit():
        print(
            "First argument must be a csv file and second/third argument must be a float.")
        print("Usage: dbscan.py <dataset> <k>")
        sys.exit(-1)
    # begin program
    wrapper(args[0], int(args[1]), int(args[2]))
