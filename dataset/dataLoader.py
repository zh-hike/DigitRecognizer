import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split


class Data(Dataset):
    def __init__(self, data_path, mode='train'):
        self.mode = mode
        self.data = None
        self.label = None
        df = pd.read_csv(os.path.join(data_path, 'train.csv')).values
        train_data, val_data = train_test_split(df, test_size=0.1, random_state=2)
        if mode == 'train':
            self.data = train_data[:, 1:].astype('float32')
            self.label = train_data[:, 0].astype('int')
        elif mode == 'val':
            self.data = val_data[:, 1:].astype('float32')
            self.label = val_data[:, 0].astype('int')
        else:
            df = pd.read_csv(os.path.join(data_path, 'test.csv')).values
            self.data = df.astype('float32')

    def __getitem__(self, item):
        if self.mode == 'train' or self.mode == 'val':
            return self.data[item], self.label[item]
        else:
            return self.data[item]

    def __len__(self):
        return self.data.shape[0]


class DL:
    def __init__(self, args):
        traindata = Data(args.data_path, 'train')
        valdata = Data(args.data_path, 'val')
        testdata = Data(args.data_path, 'test')
        self.traindl = DataLoader(traindata, batch_size=args.batch_size, shuffle=True)
        self.valdl = DataLoader(valdata, batch_size=args.batch_size, shuffle=True)
        self.testdl = DataLoader(testdata, batch_size=args.batch_size)
        