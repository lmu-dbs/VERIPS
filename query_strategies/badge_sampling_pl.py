import numpy as np
from torch.utils.data import DataLoader
from .strategy import Strategy
import pickle
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import pdb
from torch.nn import functional as F
import argparse
import torch.nn as nn
from collections import OrderedDict
from scipy import stats
import time
import numpy as np
import scipy.sparse as sp
from itertools import product
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import _num_samples
from sklearn.utils import check_array
from sklearn.utils import gen_batches
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.metrics.pairwise import rbf_kernel as rbf
from six import string_types
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from torchvision import transforms
import random

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# data perturbation (gaussian noise)
def perturb(x):
    return transforms.Compose([AddGaussianNoise(0., 5.)])(x)

# data augmentation
def augment(x):
    return nn.Sequential(transforms.RandomAffine(20, translate=(0.25, 0.25), scale=(0.75,1.25)),)(x)
    
def strong_augment(x):
    return nn.Sequential(transforms.RandomAffine(40, translate=(0.5, 0.5), scale=(0.5,1.5),shear=5),)(x)

# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distances')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    print(str(len(mu)) + '\t' + str(sum(D2)))
    return indsAll

class BadgeSamplingPL(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(BadgeSamplingPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
        self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
        self.y_pl = np.zeros(self.n_pool)

    
    # compute metrics over all pseudo-labels
    def testPL (self):
        idxs_pl = np.arange(self.n_pool)[self.idxs_pl]
        #print(idxs_pl)
        X = self.X[idxs_pl]
        #show_img(X[0])
        #show_img(X[1])
        Y = torch.Tensor(self.Y.numpy()[idxs_pl]).long()

        id = np.empty(len(X))
        id[:] = 1

        loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False, **self.args['loader_te_args'])
        inconsistency, predictions = self.measure_inconsistency(loader)

        predictions = predictions[:, :, 0]

        entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)

        return dict(zip(idxs_pl,inconsistency)), dict(zip(idxs_pl,entropy))
        
    def choosePLCandidates (self, n, idxs_unlabeled):

        X = self.X[idxs_unlabeled]
        Y = torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long()

        id = np.empty(len(X))
        id[:] = 1

        loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False, **self.args['loader_te_args'])
        inconsistency, predictions = self.measure_inconsistency(loader)

        predictions = predictions[:, :, 0]
        
        almin = np.argpartition(inconsistency, n)[:n]

        entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)
        print("EntropyPLCand " + str(np.mean(entropy[almin])))

        idxs = np.arange(self.n_pool)[idxs_unlabeled]
        self.store['scoring'][idxs] = inconsistency
        self.store['entropy'][idxs] = entropy

        return almin

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        
        gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()
        chosen = init_centers(gradEmbedding, n)
        
        return idxs_unlabeled[chosen]
    
    def queryPL(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~np.logical_or(self.idxs_lb, self.idxs_pl)]
        if len(idxs_unlabeled) >= n:
            candidates = self.choosePLCandidates(n, idxs_unlabeled)

            idxs = np.arange(self.n_pool)[idxs_unlabeled[candidates]]
            self.store['type'][idxs] = 'plcand'

            chosen = self.choosePL(idxs_unlabeled[candidates])

            idxs2 = np.arange(self.n_pool)[idxs_unlabeled[candidates[chosen]]]
            self.store['type'][idxs2] = 'plchosen'

            return idxs_unlabeled[candidates[chosen]]
        else:
            idxs = np.arange(self.n_pool)[idxs_unlabeled]
            self.store['type'][idxs] = 'plcand'
            chosen = self.choosePL(idxs_unlabeled)
            idxs2 = np.arange(self.n_pool)[idxs_unlabeled[chosen]]
            self.store['type'][idxs2] = 'plchosen'

            return idxs_unlabeled[chosen]


class BadgeSamplingRand(BadgeSamplingPL):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(BadgeSamplingRand, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
        self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
        self.y_pl = np.zeros(self.n_pool)

    def query(self, n):
        inds = np.where(self.idxs_lb == 0)[0]
        return inds[np.random.permutation(len(inds))][:n]


class BadgeSamplingRandPL(BadgeSamplingPL):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(BadgeSamplingRandPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
        self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
        self.y_pl = np.zeros(self.n_pool)
        
    def choosePLCandidates(self, n, idxs_unlabeled):
        # print(len(idxs_unlabeled))

        cand = random.sample(range(0, len(idxs_unlabeled) - 1, 1), n)

        chosen = np.array(cand)
        X = self.X[idxs_unlabeled]
        Y = torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long()
        id = np.empty(len(X))
        id[:] = 1
        loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False,
                            **self.args['loader_te_args'])
        inconsistency, predictions = self.measure_values(loader)
        predictions = predictions[:, :, 0]
        entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)
        print("EntropyPLCand " + str(np.mean(entropy[chosen])))

        idxs = np.arange(self.n_pool)[idxs_unlabeled]
        self.store['entropy'][idxs] = entropy

        return chosen


class BadgeSamplingEntropyPL(BadgeSamplingPL):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(BadgeSamplingEntropyPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
        self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
        self.y_pl = np.zeros(self.n_pool)

    def choosePLCandidates(self, n, idxs_unlabeled):
        X = self.X[idxs_unlabeled]
        Y = torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long()

        id = np.empty(len(X))
        id[:] = 1
        loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False,
                            **self.args['loader_te_args'])
        inconsistency, predictions = self.measure_values(loader)
        predictions = predictions[:, :, 0]

        entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)
        almin = np.argpartition(entropy, n)[:n]

        print("EntropyPLCand " + str(np.mean(entropy[almin])))

        idxs = np.arange(self.n_pool)[idxs_unlabeled]
        self.store['scoring'][idxs] = entropy
        self.store['entropy'][idxs] = entropy

        return almin

class BadgeSamplingMarginPL(BadgeSamplingPL):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(BadgeSamplingMarginPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
        self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
        self.y_pl = np.zeros(self.n_pool)

    def choosePLCandidates(self, n, idxs_unlabeled):
        X = self.X[idxs_unlabeled]
        Y = torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long()

        id = np.empty(len(X))
        id[:] = 1
        loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False,
                            **self.args['loader_te_args'])
        inconsistency, predictions = self.measure_values(loader)
        predictions = predictions[:, :, 0]

        predmax = np.amax(predictions, axis=1)
        predsecmax = np.sort(predictions, axis=1)[:, -2]

        margin = np.subtract(predmax, predsecmax)
        almax = np.argpartition(margin, -n)[-n:]

        entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)

        idxs = np.arange(self.n_pool)[idxs_unlabeled]
        self.store['scoring'][idxs] = margin
        self.store['entropy'][idxs] = entropy

        print("EntropyPLCand " + str(np.mean(entropy[almax])))

        return almax

def gram_red(L, L_inv, u_loc):
    n = np.shape(L_inv)[0]
    ms = np.array([False for i in range(n)])
    ms[u_loc] = True

    L_red = L[~ms][:, ~ms]

    D = L_inv[~ms][:, ~ms]
    e = L_inv[~ms][:, ms]
    f = L_inv[ms][:, ms]

    L_red_inv = D - e.dot(e.T) / f
    return L_red, L_red_inv

def gram_aug(L_Y, L_Y_inv, b_u, c_u):
    d_u = c_u - b_u.T.dot(L_Y_inv.dot(b_u))
    g_u = L_Y_inv.dot(b_u)

    L_aug = np.block([[L_Y, b_u],[b_u.T, c_u]])
    L_aug_inv = np.block([[L_Y_inv + g_u.dot(g_u.T/d_u), -g_u/d_u], [-g_u.T/d_u, 1.0/d_u]])

    return L_aug, L_aug_inv

def sample_k_imp(Phi, k, max_iter, rng=np.random):
    n = np.shape(Phi)[0]
    Ind = rng.choice(range(n), size=k, replace=False)

    if n == k:
        return Ind

    X = [False] * n
    for i in Ind:
        X[i] = True
    X = np.array(X)

    L_X = Phi[Ind, :].dot(Phi[Ind, :].T)

    L_X_inv = np.linalg.pinv(L_X)

    for i in range(1, max_iter):

        u = rng.choice(np.arange(n)[X])
        v = rng.choice(np.arange(n)[~X])

        for j in range(len(Ind)):
            if Ind[j] == u:
                u_loc = j

        L_Y, L_Y_inv = gram_red(L_X, L_X_inv, u_loc)

        Ind_red = [i for i in Ind if i != u]

        b_u = Phi[Ind_red, :].dot(Phi[[u], :].T)
        c_u = Phi[[u], :].dot(Phi[[u], :].T)
        b_v = Phi[Ind_red, :].dot(Phi[[v], :].T)
        c_v = Phi[[v], :].dot(Phi[[v], :].T)

        p = min(1, (c_v - b_v.T.dot(L_Y_inv.dot(b_v))) / (c_u - b_u.T.dot(L_Y_inv.dot(b_u))) )

        if rng.uniform() <= 1-p:
            X[u] = False
            X[v] = True
            Ind = Ind_red + [v]
            L_X, L_X_inv = gram_aug(L_Y, L_Y_inv, b_v, c_v)

        #if i % k == 0:
            # print('Iter ', i)

    return Ind

class BadgeSamplingGradientPL(BadgeSamplingPL):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(BadgeSamplingGradientPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
        self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
        self.y_pl = np.zeros(self.n_pool)
        self.filter_factor = filter_factor

    def choosePLCandidates(self, n, idxs_unlabeled):
        gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()
        chosen = np.array(sample_k_imp(gradEmbedding, n, max_iter=int(5 * n * np.log(n))))

        X = self.X[idxs_unlabeled]
        Y = torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long()
        id = np.empty(len(X))
        id[:] = 1
        loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False,
                            **self.args['loader_te_args'])
        inconsistency, predictions = self.measure_values(loader)
        predictions = predictions[:, :, 0]
        entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)
        print("EntropyPLCand " + str(np.mean(entropy[chosen])))

        idxs = np.arange(self.n_pool)[idxs_unlabeled]
        self.store['entropy'][idxs] = entropy

        return chosen


# kmeans ++ initialization
def init_centers_inv(X, K):
    ind = np.argmin([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distances')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = ((D2 ** 2)/ sum(D2 ** 2))
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), np.subtract(np.ones(len(D2)), Ddist)))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    print(str(len(mu)) + '\t' + str(sum(D2)))
    return indsAll

class BadgeSamplingGradientPL2(BadgeSamplingPL):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(BadgeSamplingGradientPL2, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
        self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
        self.y_pl = np.zeros(self.n_pool)
        self.filter_factor = filter_factor

    def choosePLCandidates(self, n, idxs_unlabeled):
        gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()
        chosen = init_centers_inv(gradEmbedding, n)

        X = self.X[idxs_unlabeled]
        Y = torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long()
        id = np.empty(len(X))
        id[:] = 1
        loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False,
                            **self.args['loader_te_args'])
        inconsistency, predictions = self.measure_values(loader)
        predictions = predictions[:, :, 0]
        entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)
        print("EntropyPLCand " + str(np.mean(entropy[chosen])))

        idxs = np.arange(self.n_pool)[idxs_unlabeled]
        self.store['entropy'][idxs] = entropy

        return chosen

class BadgeSamplingGradientPL3(BadgeSamplingPL):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(BadgeSamplingGradientPL3, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
        self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
        self.y_pl = np.zeros(self.n_pool)
        self.filter_factor = filter_factor

    def queryPL(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~np.logical_or(self.idxs_lb, self.idxs_pl)]
        if len(idxs_unlabeled) >= n:
            candidates = self.choosePLCandidates(n, idxs_unlabeled)

            idxs = np.arange(self.n_pool)[idxs_unlabeled[candidates]]
            self.store['type'][idxs] = 'plcand'

            chosen = self.choosePL(idxs_unlabeled[candidates])

            idxs2 = np.arange(self.n_pool)[idxs_unlabeled[candidates[chosen]]]
            self.store['type'][idxs2] = 'plchosen'

            return idxs_unlabeled[candidates[chosen]]
        else:
            idxs = np.arange(self.n_pool)[idxs_unlabeled]
            self.store['type'][idxs] = 'plcand'
            chosen = self.choosePL2(idxs_unlabeled)
            idxs2 = np.arange(self.n_pool)[idxs_unlabeled[chosen]]
            self.store['type'][idxs2] = 'plchosen'

            return idxs_unlabeled[chosen]

    def choosePLCandidates(self, n, idxs_unlabeled):
        gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()

        inds = np.argpartition([np.linalg.norm(s, 2) for s in gradEmbedding], n)[:n]

        X = self.X[idxs_unlabeled]
        Y = torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long()
        id = np.empty(len(X))
        id[:] = 1
        loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False,
                            **self.args['loader_te_args'])
        inconsistency, predictions = self.measure_values(loader)
        predictions = predictions[:, :, 0]
        entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)
        print("EntropyPLCand " + str(np.mean(entropy[inds])))

        idxs = np.arange(self.n_pool)[idxs_unlabeled]
        self.store['entropy'][idxs] = entropy
        self.store['scoring'][idxs] = [np.linalg.norm(s, 2) for s in gradEmbedding]

        return inds

    def choosePL2(self, idxs_unlabeled):
        gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()

        gradNorm = [np.linalg.norm(s, 2) for s in gradEmbedding]

        inds = []
        for i in range(len(idxs_unlabeled)):
            if gradNorm[i] < self.filter_factor:
                inds.append(i)

        return inds

class Margin2(BadgeSamplingPL):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(Margin2, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
        self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
        self.y_pl = np.zeros(self.n_pool)

    def choosePLCandidates(self, n, idxs_unlabeled):
        X = self.X[idxs_unlabeled]
        Y = torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long()

        id = np.empty(len(X))
        id[:] = 1
        loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False,
                            **self.args['loader_te_args'])
        inconsistency, predictions = self.measure_values(loader)
        predictions = predictions[:, :, 0]

        predmax = np.amax(predictions, axis=1)
        predsecmax = np.sort(predictions, axis=1)[:, -2]

        margin = np.subtract(predmax, predsecmax)
        almax = np.argpartition(margin, -n)[-n:]

        entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)

        idxs = np.arange(self.n_pool)[idxs_unlabeled]
        self.store['scoring'][idxs] = margin
        self.store['entropy'][idxs] = entropy

        print("EntropyPLCand " + str(np.mean(entropy[almax])))

        return almax

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        return idxs_unlabeled[U.sort()[1].numpy()[:n]]

class Entropy2(BadgeSamplingPL):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(Entropy2, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
        self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
        self.y_pl = np.zeros(self.n_pool)

    def choosePLCandidates(self, n, idxs_unlabeled):
        X = self.X[idxs_unlabeled]
        Y = torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long()

        id = np.empty(len(X))
        id[:] = 1
        loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False,
                            **self.args['loader_te_args'])
        inconsistency, predictions = self.measure_values(loader)
        predictions = predictions[:, :, 0]

        entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)
        almin = np.argpartition(entropy, n)[:n]

        print("EntropyPLCand " + str(np.mean(entropy[almin])))

        idxs = np.arange(self.n_pool)[idxs_unlabeled]
        self.store['scoring'][idxs] = entropy
        self.store['entropy'][idxs] = entropy

        return almin

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
        log_probs = torch.log(probs)
        U = (probs * log_probs).sum(1)
        return idxs_unlabeled[U.sort()[1][:n]]