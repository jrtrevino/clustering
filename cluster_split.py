import pandas as pd
import utility


def split(distance_dict, threshold, cluster_arr, master):
    dict_list = sorted(distance_dict.items(),
                       key=lambda entry: entry[1], reverse=True)
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


test = {(21, 22): 0.03511488180921566, (8, 16): 0.07408736682222543, (6, 15): 0.0924415177828127, (11, (8, 16)): 0.09255123756257283, (2, 3): 0.09571678982366998, (7, 12): 0.09773343075058494, (4, (2, 3)): 0.09838274871979523, (13, (4, (2, 3))): 0.12861907032350856, (17, 20): 0.13323450744147844, ((6, 15), (7, 12)): 0.14957289378908534, (1, (13, (4, (2, 3)))): 0.16713317757787302, (9, 14): 0.1931306845386065, ((11, (8, 16)), ((6, 15), (7, 12))): 0.24386612443022668, ((17, 20), (9, 14)): 0.27071650932084557, (24, 25): 0.2726848698432489, (23, (21, 22)): 0.2907926868977277, (10, ((17, 20), (9, 14))): 0.3033038340416547, (5, (1, (13, (4, (2, 3))))): 0.34897144631528465, (18, ((
    11, (8, 16)), ((6, 15), (7, 12)))): 0.38405887318815785, ((5, (1, (13, (4, (2, 3))))), (18, ((11, (8, 16)), ((6, 15), (7, 12))))): 0.41454039656418273, ((23, (21, 22)), (10, ((17, 20), (9, 14)))): 0.4679385232356273, (19, ((23, (21, 22)), (10, ((17, 20), (9, 14))))): 0.6266034470270837, ((24, 25), (19, ((23, (21, 22)), (10, ((17, 20), (9, 14)))))): 0.8605416490318271, (((5, (1, (13, (4, (2, 3))))), (18, ((11, (8, 16)), ((6, 15), (7, 12))))), ((24, 25), (19, ((23, (21, 22)), (10, ((17, 20), (9, 14))))))): 1.076386817640795, (0, (((5, (1, (13, (4, (2, 3))))), (18, ((11, (8, 16)), ((6, 15), (7, 12))))), ((24, 25), (19, ((23, (21, 22)), (10, ((17, 20), (9, 14)))))))): 1.0635231828186034}


df, restrictions = utility.csv_to_df('./data/mammal_milk.csv')

for value in test.copy().values():
    cluster_arr = []
    split(test.copy(), value, cluster_arr, test.copy())
    sse = 0
    for cluster in cluster_arr:
        temp_df = pd.DataFrame()
        if type(cluster) is int:
            temp_df = temp_df.append(df.loc[[cluster]])
        else:
            temp = ("{}".format(cluster).replace("(", "").replace(")", ""))
            temp = temp.split(",")
            temp = [int(x.strip()) for x in temp]
            for index in temp:
                temp_df = temp_df.append(df.loc[[index]])
            # conver temp to dataframe
            # calculate dataframe mean
            # calculate dataframe SSE
        mean = utility.get_df_mean(temp_df, restrictions, pr=False)
        sse += utility.calculate_sse([temp_df], [mean])
        print(temp_df, "\n")
    print("cluster sse: {}".format(sse))
    print("\n")

    # calculate temp_df mean
    # utility.calculate_df_mean(temp_df)
