# from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from torch.utils import data


def mean_std_table(dataframe):
    """[normalize df]

    Arguments:
        dataframe {[type]} -- [description]

    Returns:
        [dict] -- [a dictionary that contains the mean and std of every column in df]
    """
    mean_std_dict = {}
    for each_column, each_dtype in zip(iris.columns, iris.dtypes):
        if each_dtype == 'float64':
            column_mean = iris[each_column].mean()
            column_std = iris[each_column].std()
            mean_std_dict[each_column] = column_mean, column_std
            dataframe[each_column] = (
                dataframe[each_column] - column_mean)/(column_std**2)
    return mean_std_dict


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


def my_hot_encoding(dataframe, feature_list_to_encode):
    encoding_dict = {}
    for each_feature in feature_list_to_encode:
        feature_dict = {}
        for index_type, each_type in enumerate(dataframe[each_feature].unique()):
            feature_dict[each_type] = index_type
        encoding_dict[each_feature] = feature_dict
    for each_feature in feature_list_to_encode:
        dataframe[each_feature] = dataframe[each_feature].map(
            lambda x: encoding_dict[each_feature][x])
    dataframe[each_feature].astype(np.int64)
    return encoding_dict


def even_distribution(dataframe: pd, label, my_sample_number=5, shuffle=True):
    value_count_output = dataframe[label].value_counts()
    if my_sample_number <= value_count_output.min():
        unique_output_dict = value_count_output.to_dict()
        dataframe_train_list = [dataframe[dataframe[label] == each_outcome].sample(
            my_sample_number) for each_outcome in unique_output_dict]
        if shuffle:
            dataframe_train_df = pd.concat(
                dataframe_train_list).sample(frac=1).copy()
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


class FeaturesLabelSplitter(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, dataframe, feat_label_float, feat_label_int):
        """[summary]

        Arguments:
            dataframe {[type]} -- [description]
            feat_label_float {[type]} -- [description]
            feat_label_int {[type]} -- [description]
        """
        self.dataframe = dataframe
        self.feat_label_float = feat_label_float
        self.feat_label_int = feat_label_int

    def __len__(self):
        """[summary]

        Returns:
            [type] -- [description]
        """
        'Denotes the total number of samples'
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        """[summary]

        Arguments:
            index {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        'Generates one sample of data'
        # Load data and get label
        # Using torch datatypes
        X = torch.FloatTensor(
            self.dataframe[self.feat_label_float].to_numpy())[index]
        y = torch.IntTensor(
            self.dataframe[self.feat_label_int].to_numpy())[index]
        return X, y


iris_main = pd.read_csv(
    'https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv')
iris = iris_main.copy()
og_dict = mean_std_table(iris)
encoding_dict = my_hot_encoding(iris, ['Name'])
iris_train, iris_test = even_distribution(iris, 'Name', 40)
print(iris.head())
print(iris_train.head())
print(iris_test.head())
print(iris_train.shape, iris_test.shape)
print(iris.columns)
grouping_instance = FeaturesLabelSplitter(
    iris_train, ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'], ['Name'])
# print(grouping_instance.__getitem__(1))
params = {'batch_size': 7,
          'shuffle': True,
          'num_workers': 2}
training_generator = data.DataLoader(grouping_instance, **params)
# training_generator.__iter__().next()
# for each in training_generator:
#     print(each)

input_size = 4
output_size = 3

hidden1_size = 3
# hidden2_size = 32


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        # self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden1_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        # x = self.fc3(x)

        return torch.log_softmax(x, dim=-1)


model = Net()
optimizer = optim.Adam(model.parameters())
# learning rate is at default
loss_fn = nn.NLLLoss()

epochs = 500
for epoch in range(1, epochs + 1):
    for X_train_tensor, Y_train_tensor in training_generator:
        optimizer.zero_grad()
        Y_pred = model(X_train_tensor)
        loss = loss_fn(Y_pred, Y_train_tensor.type(
            torch.LongTensor).squeeze(1))
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        # print('Epoch - %d, loss - %0.2f' % (epoch, loss.item()))
        print(f'Epoch - {epoch:d}, loss - {loss.item():0.4f}')

# model.eval()
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# with torch.no_grad():

#   correct = 0
#   total = 0

#   outputs = model(x_test_tensor)
#   _, predicted = torch.max(outputs.data, 1)

#   y_test = y_test_tensor.cpu().numpy()
#   predicted = predicted.cpu()

#   print("Accuracy: ", accuracy_score(predicted, y_test))
#   print("Precision: ", precision_score(predicted, y_test, average='weighted'))
#   print("Recall: ", recall_score(predicted, y_test, average='weighted'))
