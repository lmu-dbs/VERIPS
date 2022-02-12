import numpy as np
import pdb
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.utils.data as data_utils

import glob
import os
import os.path

def get_dataset(name, path):
    if name == 'MNIST':
        return get_MNIST(path)
    elif name == 'MNISTs':
        return get_MNIST_small(path)
    elif name == 'FashionMNIST':
        return get_FashionMNIST(path)
    elif name == 'SVHN':
        return get_SVHN(path)
    elif name == 'CIFAR10':
        return get_CIFAR10(path)
    elif name == 'CIFAR10s':
        return get_CIFAR10_small(path)

def get_MNIST(path):
    raw_tr = datasets.MNIST(path + '/MNIST', train=True, download=True)
    raw_te = datasets.MNIST(path + '/MNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    #print(X_tr.numpy()[0])
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

    
def get_MNIST_small(path):
    raw_tr = datasets.MNIST(path + '/MNIST', train=True, download=True)
    raw_te = datasets.MNIST(path + '/MNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    
    np.random.seed(0)
    index = np.random.choice(X_tr.shape[0], 10000, replace=False)
    X_tr = X_tr[index]
    Y_tr = Y_tr[index]
    index2 = np.random.choice(X_te.shape[0], 2000, replace=False)
    X_te = X_te[index2]
    Y_te = Y_te[index2]
    
    return X_tr, Y_tr, X_te, Y_te

def get_FashionMNIST(path):
    raw_tr = datasets.FashionMNIST(path + '/FashionMNIST', train=True, download=True)
    raw_te = datasets.FashionMNIST(path + '/FashionMNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN(path):
    data_tr = datasets.SVHN(path + '/SVHN', split='train', download=True)
    data_te = datasets.SVHN(path +'/SVHN', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te
    
def get_CIFAR10(path):
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=True)
    
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10_small(path):
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=True)
    
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))

    startlbs = np.arange(0, 10)
    np.random.shuffle(startlbs)

    index = []

    for lb in range(0, 5):
        indices = [i for i, x in enumerate(Y_tr) if x == startlbs[lb]]
        index.extend(indices)

    for lb in range(5, 10):
        indices = [i for i, x in enumerate(Y_tr) if x == startlbs[lb]]
        chosen = np.random.choice(indices, 500, replace=False)
        index.extend(chosen)

    X_tr2 = X_tr[index]
    Y_tr2 = Y_tr[index]

    print(X_tr2.shape)
    print(np.bincount(Y_tr2))

    return X_tr2, Y_tr2, X_te, Y_te

def get_handler(name):
    if name == 'MNIST':
        return DataHandler3
    elif name == 'MNISTs':
        return DataHandler3
    elif name == 'FashionMNIST':
        return DataHandler1
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3
    elif name == 'CIFAR10s':
        return DataHandler3
    else:
        return DataHandler4

class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.id = id

    def __getitem__(self, index):
        x, y, id = self.X[index], self.Y[index], self.id[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, id, index

    def __len__(self):
        return len(self.X)

class DataHandler2(Dataset):
    def __init__(self, X, Y, id, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.id = id

    def __getitem__(self, index):
        x, y, id = self.X[index], self.Y[index], self.id[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, id, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
    def __init__(self, X, Y, id, transform=None,):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.id = id

    def __getitem__(self, index):
        x, y, id = self.X[index], self.Y[index], self.id[index]

        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, id, index

    def __len__(self):
        return len(self.X)

class DataHandler4(Dataset):
    def __init__(self, X, Y, id, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.id = id

    def __getitem__(self, index):
        x, y, id = self.X[index], self.Y[index], self.id[index]
        return x, y, id, index

    def __len__(self):
        return len(self.X)
