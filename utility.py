import pandas as pd


def csv_to_df(path_to_csv):
    counter = 0
    with open(path_to_csv, "r") as file:
        restrictions = list(map(int, file.readline().rstrip().split(',')))
    headers = get_headers(path_to_csv)
    df = pd.read_csv(path_to_csv, skiprows=[0], names=headers)
    for column in df:
        print(column)
        if restrictions[counter] == 1:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        counter += 1
    return df, restrictions


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


df, restrictions = csv_to_df('./data/AccidentsSet03.csv')
print(df)
