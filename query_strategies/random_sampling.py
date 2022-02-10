import numpy as np
from .strategy import Strategy
import pdb

class RandomSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        super(RandomSampling, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)

    def query(self, n):
        inds = np.where(self.idxs_lb==0)[0]
        return inds[np.random.permutation(len(inds))][:n]

    def choosePLCandidates(self, n, idxs_unlabeled):
        return []


    def queryPL(self, n):
        return []