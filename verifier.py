import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.optim as optim
import pdb
from torch.nn import functional as F
import torch.nn as nn
from scipy import stats
from sklearn.metrics import pairwise_distances
from torchvision import transforms
import random

from query_strategies import BadgeSampling, BadgeSamplingPL, BALDDropout

# Verifier architecture

class Verifier:
    def __init__(self, n_pool, args, idxs_pl, pl_lbs):
        self.args = args
        self.predTest_pl = np.zeros(n_pool)
        self.idxs_pl = idxs_pl
        self.pl_lbs = pl_lbs

    # add pseudo-labeled/labeled data

    def updatePL(self, idxs_lb, idxs_pl, pl_lbs, pl_idxs_store):
        pass

    # train network(s)

    def train(self, noearlystop, epochnum):
        pass

    # predict with network

    def predict(self, X, Y):
        pass

    # verifiy the pseudo-labels, returns boolean if verified

    def verify(self, pl_idxs_store, pred_pl):
        pass

# Basic Verifier setup, badge based

class BasicVerifier(Verifier):
    def __init__(self, X, Y, idxs_lb, net, handler, n_pool, args, idxs_pl, pl_lbs):
        super(BasicVerifier, self).__init__(n_pool, args, idxs_pl, pl_lbs)
        self.model = BadgeSampling(X, Y, idxs_lb, net, handler, args)
        self.X = X
        self.Y = Y

    #def update(self, idxs_lb):
    #    self.model.update(idxs_lb)

    def updatePl(self, idxs_lb, idxs_pl, pl_lbs, pl_idxs_store):
        self.model.update(idxs_lb)
        self.pl_lbs = pl_lbs

    def train(self, noearlystop, epochnum):
        self.model.train(noearlystop, epochnum)

    def predict(self, X, Y):
        return self.model.predict(X, Y)

    def verify(self, pl_idxs_store, pred_pl):
        self.predTest_pl[pl_idxs_store] = (
            self.model.predict(self.X[pl_idxs_store], torch.Tensor(self.Y.numpy()[pl_idxs_store]).long())).numpy()
        return self.predTest_pl[pl_idxs_store] == pred_pl[pl_idxs_store]

# Basic verifier, BALD based

class BALDVerifier(BasicVerifier):
    def __init__(self, X, Y, idxs_lb, net, handler, n_pool, args, idxs_pl, pl_lbs):
        super(BALDVerifier, self).__init__(X, Y, idxs_lb, net, handler, n_pool, args, idxs_pl, pl_lbs)
        self.model = BALDDropout(X, Y, idxs_lb, net, handler, args, 1, 'marginratio')
        self.X = X
        self.Y = Y


# Verifier for pseudo-label-based verification, BADGE-based

class PLVerifier(Verifier):
    def __init__(self, X, Y, idxs_lb, net, net2, handler, n_pool, args, idxs_pl, pl_lbs, filter_factor):
        super(PLVerifier, self).__init__(n_pool, args, idxs_pl, pl_lbs)
        self.idxs_lb = idxs_lb
        self.modelpl = BadgeSamplingPL(X, Y, idxs_lb, net, handler, args, filter_factor, 'marginratio')
        self.pl_idxs_store = []
        self.verifier = BasicVerifier(X, Y, idxs_lb, net2, handler, n_pool, args, [], pl_lbs)
        self.X = X
        self.Y = Y

    def updatePl(self, idxs_lb, idxs_pl, pl_lbs, pl_idxs_store):
        self.modelpl.update(idxs_lb)
        self.idxs_lb = idxs_lb
        self.verifier.updatePL(idxs_pl, pl_lbs)
        self.pl_idxs_store = pl_idxs_store
        self.idxs_pl = idxs_pl
        self.pl_lbs = pl_lbs

    def train(self, noearlystop, epochnum):
        self.verifier.train(noearlystop, epochnum)

        verification = dict(zip(self.pl_idxs_store, self.verifier.verify(self.pl_idxs_store, self.pl_lbs)))

        for key in verification:
            if not verification:
                self.idxs_pl[key] = False

        self.modelpl.updatepl(self.idxs_lb, self.idxs_pl, self.pl_lbs)
        self.modelpl.trainPL(1, False, noearlystop, epochnum)

    def predict(self, X, Y):
        P_test = self.verifier.predict(X, Y)
        acctest = 1.0 * (Y == P_test).sum().item() / len(Y)
        print("Internal Test: " + '\t' + 'testing accuracy {}'.format(acctest), flush=True)
        return self.modelpl.predict(X, Y)

    def verify(self, pl_idxs_store, pred_pl):

        self.predTest_pl[pl_idxs_store] = (
            self.modelpl.predict(self.X[pl_idxs_store], torch.Tensor(self.Y.numpy()[pl_idxs_store]).long())).numpy()
        return self.predTest_pl[pl_idxs_store] == pred_pl[pl_idxs_store]
