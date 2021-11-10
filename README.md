# Unsupervised Learning

## Members
Joey Trevino jrtrevin@calpoly.edu
David Nguyen dnguy260@calpoly.edu

## Dependencies
For drawing clusters, we used matplotlib.pyplot. To install this, type 
```
python -m pip install -U matplotlib
```

### k-means
To run this program, invoke it with the following command:
```
python3 kmeans.py <dataset.csv> <k-value>
```

This will run the clustering with an initial cluster set of size k. Cluters are chosen by sampling the dataset, determining the mean, and calculating the furthest point away from the mean. Subsequent clusters are found by maximizing the distance between a point and previously generated clusters.


### hierarchical 
To run this program, invoke it with the following command:
```
python3 hclustering.py <dataset.csv> [threshold]
```
Outputs: 
1. json file containing tree
2. if given a threshold, will produce a pdf containing clusters and their points


### dbscan
To run this program, type
```
python3 dbscan.py <min_points> <epsilon>
```
This program uses euclidean distance to cluster datapoints together. Data is determined to be either a:
        1. Core point (more than min_points in a neighborhood)
        2. Boundary point (less than min_points but a neighbor is a core point)
        3. Noise (above two conditions fail)

DBSCAN returns an array of dataframes representing clusters, and a dataframe representing the outliers.
We chose not to graph outliers for this assignment, but data calculations for these points can be found by running the program or viewing the appendix section of the report.
