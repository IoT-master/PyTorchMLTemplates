from torch.utils import data
from pandas import pd

class FeaturesLabelSplitter(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataframe, features, labels):
        'Initialization'
        super(FeaturesLabelSplitter, self).__init__()
        self.dataframe = dataframe
        self.labels = labels
        self.list_IDs = features

    def __len__(self):
        'Denotes the total number of samples'
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = torch.FloatTensor(self.dataframe[self.list_IDs].to_numpy())[index]
        y = (torch.LongTensor(self.dataframe[self.labels].to_numpy())).squeeze(1)[index]

        return X, y

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

def train_test_split_even(dataframe, label_name, shuffle=True, test_decimal=0.2, even_num_labels=True):
    assert test_decimal < 1
    assert test_decimal > 0
    tt_boarder = int(dataframe.shape[0]*(1-test_decimal))
    if even_num_labels:
        min_sample = min(y_train.value_counts())
        
        train_df = pd.DataFrame(columns=dataframe.columns)
        if tt_boarder >= min_sample*len(type_count)*(1-test_decimal):
            if shuffle:
                for each_type in dataframe[label_name].unique():
                    train_df = train_df.append(dataframe[each_type == dataframe[label_name]].sample(frac=(1-test_decimal)))     
            else:
                for each_type in dataframe[label_name].unique():
                    train_df = train_df.append(dataframe[each_type == dataframe[label_name]][:tt_boarder//len(type_count)])
        else:
            if shuffle:
                for each_type in dataframe[label_name].unique():
                    train_df = train_df.append(dataframe[each_type == dataframe[label_name]].sample(frac=(1-test_decimal)))
            else:
                for each_type in dataframe[label_name].unique():
                    train_df = train_df.append(dataframe[each_type == dataframe[label_name]][:tt_boarder//len(type_count)])
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
        dataframe[each_feature] = dataframe[each_feature].map(lambda x: encoding_dict[each_feature][x])
    return encoding_dict