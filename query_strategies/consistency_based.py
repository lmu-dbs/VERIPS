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


# data augmentation
def augment(x):
    return nn.Sequential(transforms.RandomAffine(20, translate=(0.25, 0.25), scale=(0.75, 1.25)), )(x)


class ConsistencyBased(Strategy):

    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(ConsistencyBased, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
        self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
        self.y_pl = np.zeros(self.n_pool)

    # update for pseudo-labels
    def updatepl(self, idxs_lb, idxs_pl, y_pl):
        self.idxs_lb = idxs_lb
        self.idxs_pl = idxs_pl
        self.y_pl = y_pl

    # choose samples to send to oracle
    def chooseAL(self, n, idxs_unlabeled):

        X = self.X[idxs_unlabeled]
        Y = torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long()

        id = np.empty(len(X))
        id[:] = 1

        loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False,
                            **self.args['loader_te_args'])
        inconsistency, predictions = self.measure_inconsistency(loader)

        almax = np.argpartition(inconsistency, -n)[-n:]

        return almax

    # choose samples to send to oracle and pseudo-label candidates
    def choosePLCandidates(self, n, idxs_unlabeled):

        X = self.X[idxs_unlabeled]
        Y = torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long()

        id = np.empty(len(X))
        id[:] = 1

        loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False,
                            **self.args['loader_te_args'])
        inconsistency, predictions = self.measure_inconsistency(loader)

        # n minimum elements based on inconsistency
        almin = np.argpartition(inconsistency, n)[:n]

        predictions = predictions[:, :, 0]

        entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)
        print("EntropyPLCand " + str(np.mean(entropy[almin])))

        return almin

    # compute metrics over all pseudo-labels
    def testPL(self):
        idxs_pl = np.arange(self.n_pool)[self.idxs_pl]
        # print(idxs_pl)
        X = self.X[idxs_pl]
        # show_img(X[0])
        # show_img(X[1])
        Y = torch.Tensor(self.Y.numpy()[idxs_pl]).long()

        id = np.empty(len(X))
        id[:] = 1

        loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False,
                            **self.args['loader_te_args'])
        inconsistency, predictions = self.measure_inconsistency(loader)

        predictions = predictions[:, :, 0]

        entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)

        return dict(zip(idxs_pl, inconsistency)), dict(zip(idxs_pl, entropy))

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        chosen = self.chooseAL(n, idxs_unlabeled),

        return idxs_unlabeled[chosen]

    def queryPL(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~np.logical_or(self.idxs_lb, self.idxs_pl)]
        if len(idxs_unlabeled) >= n:
            candidates = self.choosePLCandidates(n, idxs_unlabeled)
            chosen = self.choosePL(idxs_unlabeled[candidates])
            return idxs_unlabeled[candidates[chosen]]
        else:
            return idxs_unlabeled[[]]

    def train(self):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        n_epoch = self.args['n_epoch']
        self.clf = self.net.apply(weight_reset).cuda()
        optimizer = optim.Adam(self.clf.parameters(), lr=self.args['lr'], weight_decay=0)

        idxs_train_lb = np.arange(self.n_pool)[self.idxs_lb]
        idxs_train_ul = np.arange(self.n_pool)[~self.idxs_lb]

        id_lb = np.empty(len(idxs_train_lb))
        id_lb[:] = 1

        id_ul = np.empty(len(idxs_train_ul))
        id_ul[:] = 3

        X_dl = np.concatenate([self.X[idxs_train_lb], self.X[idxs_train_ul]], axis=0)
        Y_dl = np.concatenate([self.Y.numpy()[idxs_train_lb], self.Y.numpy()[idxs_train_ul]], axis=0)
        id_dl = np.concatenate([id_lb, id_ul], axis=0)

        # X_dl = np.concatenate([self.X[idxs_train_lb]], axis=0)
        # Y_dl = np.concatenate([self.Y.numpy()[idxs_train_lb]], axis=0)
        # id_dl = np.concatenate([id_lb], axis=0)

        lbpl = len(idxs_train_lb)
        # print("comb: " + str(lbpl))
        # print(id_dl)

        loader_tr = DataLoader(self.handler(X_dl, torch.Tensor(Y_dl).long(), id_dl, transform=self.args['transform']),
                               shuffle=True, **self.args['loader_tr_args'])

        lb_weight = 1

        ul_weight = 1 / len(idxs_train_ul)

        epoch = 0
        accCurrent = 0.
        while accCurrent < 0.99 and epoch < 50:
            accCurrent = self._trainPL(epoch, loader_tr, optimizer, 0, lb_weight, ul_weight) / len(idxs_train_lb)
            print("Epoch: " + str(epoch) + " " + str(accCurrent))
            epoch += 1
            #if (epoch % 50 == 0) and (accCurrent < 0.2):  # reset if not converging
            #    self.clf = self.net.apply(weight_reset)
            #    optimizer = optim.Adam(self.clf.parameters(), lr=self.args['lr'], weight_decay=0)
        print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)

    def _trainPL(self, epoch, loader_tr, optimizer, pl_factor, lb_weight, ul_weight):
        self.clf.train()
        clb = 0
        cpl = 0
        cul = 0

        switch = False

        for batch_idx, (x, y, id, idxs) in enumerate(loader_tr):

            x_lb = x[id == 1]
            y_lb = y[id == 1]
            x_pl = x[id == 2]
            y_pl = y[id == 2]
            x_ul = x[id == 3]
            y_ul = y[id == 3]

            clb += len(x_lb)
            cpl += len(x_pl)
            cul += len(x_ul)
            if (len(x_lb) > 1):
                if (len(x_lb) == 1):
                    switch = True
                    self.clf.eval()

                x_lb, y_lb = Variable(x_lb.cuda()), Variable(y_lb.cuda())
                optimizer.zero_grad()
                out_lb, e1 = self.clf(x_lb)
                loss_lb = F.cross_entropy(out_lb, y_lb)
                #accFinal += torch.sum((torch.max(out_lb, 1)[1] == y_lb).float()).data.item()

                if (switch):
                    self.clf.train()
                    switch = False

            else:
                loss_lb = 0

            if (len(x_pl) > 1):
                if (len(x_pl) == 1):
                    switch = True
                    self.clf.eval()

                x_pl, y_pl = Variable(x_pl.cuda()), Variable(y_pl.cuda())
                optimizer.zero_grad()
                out_pl, e2 = self.clf(x_pl)
                loss_pl = F.cross_entropy(out_pl, y_pl)
                #accFinal += torch.sum((torch.max(out_pl, 1)[1] == y_pl).float()).data.item()

                if (switch):
                    self.clf.train()
                    switch = False
            else:
                loss_pl = 0

            if (len(x_ul) > 1):

                if (len(x_ul) == 1):
                    switch = True
                    self.clf.eval()

                x_aug = augment(x_ul)

                x_ul, y_ul = Variable(x_ul.cuda()), Variable(y_ul.cuda())
                x_aug = Variable(x_aug.cuda())
                optimizer.zero_grad()

                outu, eu = self.clf(x_ul)
                probu = F.softmax(outu, dim=1)

                outa, ea = self.clf(x_aug)
                proba = F.softmax(outa, dim=1)

                # D in Lu(x,M)

                loss_ul = F.kl_div(probu, proba, reduction='sum')

                # if (batch_idx == 1):
                # print("probu: " + str(probu))
                # print("proba: " + str(proba))
                # print("loss_ul: " + str(loss_ul) + " " + str(ul_weight* loss_ul))
                if (switch):
                    self.clf.train()
                    switch = False

            else:
                loss_ul = 0

            loss = lb_weight * loss_lb + pl_factor * loss_pl + ul_weight * loss_ul

            # if (batch_idx == 1):
            # print("loss_lb: " + str(loss_lb) + " " + str(lb_weight * loss_lb))
            # print("loss: " + str(loss))
            # print("----")

            loss.backward()


            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
        # print("clb: " + str(clb) + " cpl: " + str(cpl) + " cul: " + str(cul))
        self.clf.eval()
        accFinal = 0.
        clb = 0
        for batch_idx, (x, y, id, idxs) in enumerate(loader_tr):

            x_lb = x[id == 1]
            y_lb = y[id == 1]

            clb += len(x_lb)
            if (len(x_lb) > 0):
                x_lb, y_lb = Variable(x_lb.cuda()), Variable(y_lb.cuda())
                out_lb, e1 = self.clf(x_lb)
                accFinal += torch.sum((torch.max(out_lb, 1)[1] == y_lb).float()).data.item()

        return accFinal

    def trainPL(self, pl_factor, pl_ratio):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        n_epoch = self.args['n_epoch']
        self.clf = self.net.apply(weight_reset).cuda()
        optimizer = optim.Adam(self.clf.parameters(), lr=self.args['lr'], weight_decay=0)

        idxs_train_lb = np.arange(self.n_pool)[self.idxs_lb]
        idxs_train_pl = np.arange(self.n_pool)[self.idxs_pl]
        idxs_train_ul = np.arange(self.n_pool)[~self.idxs_lb]

        id_lb = np.empty(len(idxs_train_lb))
        id_lb[:] = 1

        id_pl = np.empty(len(idxs_train_pl))
        id_pl[:] = 2

        id_ul = np.empty(len(idxs_train_ul))
        id_ul[:] = 3

        X_dl = np.concatenate([self.X[idxs_train_lb], self.X[idxs_train_pl], self.X[idxs_train_ul]], axis=0)
        Y_dl = np.concatenate([self.Y.numpy()[idxs_train_lb], self.y_pl[idxs_train_pl], self.Y.numpy()[idxs_train_ul]],
                              axis=0)
        id_dl = np.concatenate([id_lb, id_pl, id_ul], axis=0)

        # X_dl = np.concatenate([self.X[idxs_train_lb]], axis=0)
        # Y_dl = np.concatenate([self.Y.numpy()[idxs_train_lb]],axis=0)
        # id_dl = np.concatenate([id_lb], axis=0)

        # print("comb: " + str(lbpl))
        # print(id_dl)

        loader_tr = DataLoader(self.handler(X_dl, torch.Tensor(Y_dl).long(), id_dl, transform=self.args['transform']),
                               shuffle=True, **self.args['loader_tr_args'])
        lb_weight = 1

        ul_weight = 1 / len(idxs_train_ul)

        ratiopl = 1
        if(pl_ratio):
            ratiopl = len(idxs_train_pl)/len(idxs_train_lb)
        plfactor = pl_factor * ratiopl

        epoch = 0
        accCurrent = 0.
        while accCurrent < 0.99 and epoch < 50:
            accCurrent = self._trainPL(epoch, loader_tr, optimizer, pl_factor, lb_weight, ul_weight) / len(idxs_train_lb)
            epoch += 1
            print("Epoch: " + str(epoch) + " " + str(accCurrent))
            if (epoch % 50 == 0) and (accCurrent < 0.2):  # reset if not converging
                self.clf = self.net.apply(weight_reset)
                optimizer = optim.Adam(self.clf.parameters(), lr=self.args['lr'], weight_decay=0)
        print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)
        return accCurrent

class ConsistencyBasedRand(ConsistencyBased):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(ConsistencyBasedRand, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
        self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
        self.y_pl = np.zeros(self.n_pool)

    def query(self, n):
        inds = np.where(self.idxs_lb==0)[0]
        return inds[np.random.permutation(len(inds))][:n]

class ConsistencyBasedRandPL(ConsistencyBased):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(ConsistencyBasedRandPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
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

        return chosen

class ConsistencyBasedEntropyPL(ConsistencyBased):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(ConsistencyBasedEntropyPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
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

        return almin

class ConsistencyBasedMarginPL(ConsistencyBased):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(ConsistencyBasedMarginPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
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

class ConsistencyBasedGradientPL(ConsistencyBased):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(ConsistencyBasedGradientPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
        self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
        self.y_pl = np.zeros(self.n_pool)

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

        return chosen