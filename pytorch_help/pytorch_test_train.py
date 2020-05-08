import torch
import pandas as pd
import numpy as np


def even_distribution(dataframe: pd, label, my_sample_number=5, shuffle=True):
    value_count_output = dataframe[label].value_counts()
    if my_sample_number <= value_count_output.min():
        unique_output_dict = value_count_output.to_dict()
        dataframe_train_list = [dataframe[dataframe[label] == each_outcome].sample(my_sample_number) for each_outcome in unique_output_dict]
        if shuffle:
            dataframe_train_df = pd.concat(dataframe_train_list).sample(frac=1).copy()
        else:
            dataframe_train_df = pd.concat(dataframe_train_list).copy()
        dataframe_test_df = dataframe.drop(dataframe_train_df.index).copy()
        return dataframe_train_df, dataframe_test_df
    else:
        print(value_count_output)
        print(dataframe.shape)
        # print('Resorting to normal output')
        assert my_sample_number > dataframe.shape[0]
        if shuffle:
            dataframe_train_df = dataframe.sample(my_sample_number).copy()
        else:
            dataframe_train_df = dataframe.head(my_sample_number).copy()
        return dataframe_train_df, dataframe.drop(dataframe_train_df.index).copy()

def my_hot_encoding(dataframe, feature_list_to_encode):
    encoding_dict = {}
    for each_feature in feature_list_to_encode:
        feature_dict = {}
        for index_type, each_type in enumerate(dataframe[each_feature].unique()):
            feature_dict[each_type] = index_type
        encoding_dict[each_feature] = feature_dict
    for each_feature in feature_list_to_encode:
        dataframe[each_feature] = dataframe[each_feature].map(lambda x: encoding_dict[each_feature][x])
    dataframe[each_feature].astype(np.int64)
    return encoding_dict        

def train_test_split(dataframe, shuffle=False, test_decimal=.2):
    assert test_decimal < 1
    assert test_decimal > 0
    if shuffle:
        train_df = dataframe.sample(frac=(1-test_decimal))
    else:
        tt_boarder = int(dataframe.shape[0]*(1-test_decimal))
        train_df = dataframe[:tt_boarder]
    test_df = dataframe.drop(train_df.index)
    return train_df, test_df

def mean_std_table(dataframe):
    mean_std_dict = {}
    for each_column, each_dtype in zip(iris.columns, iris.dtypes):
        if each_dtype == 'float64':
            column_mean = iris[each_column].mean()
            column_std = iris[each_column].std()
            mean_std_dict[each_column] = column_mean, column_std
            dataframe[each_column] = (dataframe[each_column]- column_mean)/(column_std**2)
    return mean_std_dict

if __name__=="__main__":
    iris_main = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv')
    iris = iris_main.copy()
    x, y = even_distribution(iris, 'Name', 30, True)
    print(x, len(x))
    print(y, len(y))
    

    # unique_dict = my_hot_encoding(iris, ['Name'])
    # print(iris.dtypes)
    # print(unique_dict)
    # print(iris)
    # print(iris.dtypes)

    # a, b = train_test_split(iris)
    # print(a)
    # print(b)

    # my_stats_dict = mean_std_table(iris)
    # print(my_stats_dict)