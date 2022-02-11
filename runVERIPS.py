import numpy as np
import sys
import openml
import os
from dataset import get_dataset, get_handler
import vgg
import resnet
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torch
import time
import random
import math
import collections

from query_strategies import BadgeSamplingPL, BadgeSamplingRandPL, \
    BadgeSamplingEntropyPL, BadgeSamplingMarginPL, \
    BadgeSamplingGradientPL, BadgeSamplingRand, BALDDropout, BALDDropoutRand, BALDDropoutRandPL, \
    BALDDropoutEntropyPL, BALDDropoutMarginPL, BALDDropoutGradientPL, BALDDropoutConsistencyPL, \
    BadgeSamplingGradientPL2, BadgeSamplingGradientPL3, \
    BALDDropoutPL3, Margin2, Entropy2, BALDDropoutBALDInv

from verifier import BasicVerifier, PLVerifier, BALDVerifier

# arguments

# based off Batch Active learning by Diverse Gradient Embeddings (BADGE) by JordanAsh and Chicheng Zhang
# which was originally built by modifying Kuan-Hao Huang's deep active learning repository.

# nStart: number of labeled data points at the start
# nEnd: max number of labeled data points at the end
# nQuery: number of new labeled data points added each timestep
# nPseudo: number of new pseudo-labels added each timestep
# veri: verification method
#       0: no verification
#       1: perfect verification (test only)
#       2: verification network with labeled data
#       3: verification with network with both pseudo labels and labeled data
#       4: early label-based verification
# filter: filter factor
# meta: 0 no meta pseudo labels, 1 pseudo label labels generated based on verifier
# alg: baseline algorithm
#       'badge': BADGE
#       'bald': BALD
# plmethod:
#       'none': no pseudo labels,
#       'consistency': consistency-based PL,
#       'gradient': gram-based PL,
#       'gradient3': gradient-based PL
#       'entropy': entropy-based PL,
#       'margin': margin-based PL
#       'random': random PL
# filetrtype: filter method, (filter value given in 'filter', decay in 'filterdecay')
#       'entropy': entropy-based filter
#       'margintrue': margin-based filter
#       'margin': margin-ratio-based filter
# loss: loss method
#       'fixed': constant value for pl weight (defined by param)
#       'timestep': timestep value for pl weight
#       'entropy': pl weight based on inverse of overall entropy times param
#       'overlap': pl weight based on overlap with verifier times param
#       'accuracy': pl weight based on prediction accuracy times param
# lossparam: defining parameter for loss method
# lossmin: min loss
# lossmax: max loss
# start: starting method, only supported method is 'step' which uses a fixed starting timestep given in 'startparam'
# startdata: handling of starting data
# remove: use pseudo-label dropping
# plratio: use ratio-based weighting
# noearlystop: don't use a fixed training stopping point (otherwise given in 'epochnum')
# dual: retrain the model prior to pseudo-label selection
# replenish: try to always maintaining the maximal allowed number of pseudo-labels

opts = {'model': 'vgg', 'nQuery': 1000, 'nPseudo': 1000000, 'lr': 0.001, 'data': 'CIFAR10', 'alg': 'entropy',
        'plmethod': 'entropy', 'C': 'margin', 'dual': False, 'epochnum': 100, 'nClasses': 10, 'startdata': "base",
        'path': 'data', 'nStart': 100, 'nEnd': 10100, 'nEmb': 256, 'did': 0, 'meta': 0, 'filtertype': 'margin',
        'veri': 0, 'veriparam': 1, 'filter': 0, 'filterdecay':0, 'loss': 'fixed', 'lossparam': 1, 'lossmin': 0, 'lossmax': 1,
        'start': 'step', 'startparam': 0, 'remove': False, 'plratio': False, 'noearlystop': False, 'replenish': False}

veristr = {0: "no_veri", 1: "perf", 2: "valilb", 3: "valiul", 4:"valiearly"}


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False


# %%

def calc_entropy(predictions):
    entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)
    return np.mean(entropy)

def calc_margin(predictions):
    probmaxs = np.amax(predictions, axis=1)
    probsecmaxs = np.sort(predictions, axis=1)[:, -2]

    margin =  np.subtract(probmaxs, probsecmaxs)
    return np.mean(margin)



def main(log_file, opts, runPrefix, runname):
    # code based on https://github.com/ej0cl6/deep-active-learning"

    NUM_INIT_LB = opts['nStart']
    NUM_QUERY = opts['nQuery']
    NUM_PSEUDO = opts['nPseudo']
    NUM_ROUND = int((opts['nEnd'] - NUM_INIT_LB) / NUM_QUERY)
    DATA_NAME = opts['data']

    #print(run_id)
    print(NUM_ROUND)
    print(opts)

    inconsistency_store = dict()
    entropy_store = dict()
    correctness_store = dict()
    verification_store = dict()

    # non-openml data defaults
    args_pool = {'MNIST':
                     {'n_epoch': 10, 'transform': transforms.Compose(
                         [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                      'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
                      'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                      'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
                 'MNISTs':
                     {'n_epoch': 10, 'transform': transforms.Compose(
                         [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                      'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
                      'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                      'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
                 'FashionMNIST':
                     {'n_epoch': 10, 'transform': transforms.Compose(
                         [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))]),
                      'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
                      'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                      'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
                 'CIFAR10':
                     {'n_epoch': 3, 'transform': transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                          (0.2470, 0.2435, 0.2616))]),
                      'loader_tr_args': {'batch_size': 128, 'num_workers': 1},
                      'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                      'optimizer_args': {'lr': 0.05, 'momentum': 0.3},
                      'transformTest': transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                (0.2470, 0.2435, 0.2616))])}
                 }
    args_pool['CIFAR10'] = {'n_epoch': 3,
                            'transform': transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                  (0.2470, 0.2435, 0.2616))]),
                            'loader_tr_args': {'batch_size': 128, 'num_workers': 1},
                            'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                            'optimizer_args': {'lr': 0.05, 'momentum': 0.3},
                            'transformTest': transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                      (0.2470, 0.2435, 0.2616))])
                            }
    args_pool['CIFAR10s'] = {'n_epoch': 3,
                             'transform': transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                   (0.2470, 0.2435, 0.2616))]),
                             'loader_tr_args': {'batch_size': 128, 'num_workers': 1},
                             'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                             'optimizer_args': {'lr': 0.05, 'momentum': 0.3},
                             'transformTest': transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                       (0.2470, 0.2435, 0.2616))])
                             }

    args_pool['SVHN'] = {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}}

    args_pool['MNIST'] = {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]),
                      'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
                      'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                      'optimizer_args': {'lr': 0.01, 'momentum': 0.5}}


    args_pool['CIFAR10']['transform'] = args_pool['CIFAR10']['transformTest']  # remove data augmentation
    args_pool['CIFAR10s']['transform'] = args_pool['CIFAR10s']['transformTest']
    args_pool['MNISTs']['transformTest'] = args_pool['MNISTs']['transform']
    args_pool['MNIST']['transformTest'] = args_pool['MNIST']['transform']
    args_pool['SVHN']['transformTest'] = args_pool['SVHN']['transform']

    if opts['did'] == 0: args = args_pool[DATA_NAME]
    if not os.path.exists(opts['path']):
        os.makedirs(opts['path'])


    # load openml dataset if did is supplied
    if opts['did'] > 0:
        openml.config.apikey = '3411e20aff621cc890bf403f104ac4bc'
        openml.config.set_cache_directory(opts['path'])
        ds = openml.datasets.get_dataset(opts['did'])
        data = ds.get_data(target=ds.default_target_attribute)
        X = np.asarray(data[0])
        y = np.asarray(data[1])
        y = LabelEncoder().fit(y).transform(y)

        opts['nClasses'] = int(max(y) + 1)
        nSamps, opts['dim'] = np.shape(X)
        testSplit = .1
        inds = np.random.permutation(nSamps)
        X = X[inds]
        y = y[inds]

        split = int((1. - testSplit) * nSamps)
        while True:
            inds = np.random.permutation(split)
            if len(inds) > 50000: inds = inds[:50000]
            X_tr = X[:split]
            X_tr = X_tr[inds]
            X_tr = torch.Tensor(X_tr)

            y_tr = y[:split]
            y_tr = y_tr[inds]
            Y_tr = torch.Tensor(y_tr).long()

            X_te = torch.Tensor(X[split:])
            Y_te = torch.Tensor(y[split:]).long()

            if len(np.unique(Y_tr)) == opts['nClasses']: break

        args = {'transform': transforms.Compose([transforms.ToTensor()]),
                'n_epoch': 10,
                'loader_tr_args': {'batch_size': 128, 'num_workers': 1},
                'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                'optimizer_args': {'lr': 0.01, 'momentum': 0},
                'transformTest': transforms.Compose([transforms.ToTensor()])}
        handler = get_handler('other')

    # load non-openml dataset
    else:
        X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, opts['path'])
        opts['dim'] = np.shape(X_tr)[1:]
        print("dim: " + str(opts['dim']))
        handler = get_handler(opts['data'])

        print("Training X Shape " + str(X_tr.shape))
        print("Training Y Shape " + str(Y_tr.shape))
        print("Testing X Shape " + str(X_te.shape))
        print("Testing Y Shape " + str(Y_te.shape))

    args['lr'] = opts['lr']

    # start experiment
    n_pool = len(Y_tr)
    n_test = len(Y_te)

    print('number of pool: {}'.format(n_pool), flush=True)
    print('number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
    print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
    print('number of testing pool: {}'.format(n_test), flush=True)

    # generate initial labeled pool
    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_pl = np.zeros(n_pool, dtype=bool)

    if(opts['startdata'] == 'equal'):
        num_lb = math.ceil((NUM_INIT_LB) / opts['nClasses'])
        for lb in range(0, opts['nClasses']):
            indices = [i for i, x in enumerate(Y_tr) if x == lb]
            np.random.shuffle(indices)
            idxs_lb[indices[:num_lb]] = True
    elif (opts['startdata'] == 'oneout'):

        num_lb = math.ceil((NUM_INIT_LB) / (opts['nClasses'] - 1))

        startlbs = np.arange(0, opts['nClasses'])
        np.random.shuffle(startlbs)

        for lb in range(0, opts['nClasses'] - 1):
            indices = [i for i, x in enumerate(Y_tr) if x == startlbs[lb]]
            np.random.shuffle(indices)
            idxs_lb[indices[:num_lb]] = True
    elif (opts['startdata'] == 'imbalance'):

        num_lb = math.ceil((NUM_INIT_LB) / (opts['nClasses'] - 5))

        startlbs = np.arange(0, opts['nClasses'])
        np.random.shuffle(startlbs)

        for lb in range(0, opts['nClasses'] - 5):
            indices = [i for i, x in enumerate(Y_tr) if x == startlbs[lb]]
            np.random.shuffle(indices)
            idxs_lb[indices[:num_lb]] = True

    else:
        idxs_tmp = np.arange(n_pool)
        np.random.shuffle(idxs_tmp)
        idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

    # linear model class
    class linMod(nn.Module):
        def __init__(self, nc=1, sz=28):
            super(linMod, self).__init__()
            self.lm = nn.Linear(int(np.prod(opts['dim'])), opts['nClasses'])

        def forward(self, x):
            x = x.view(-1, int(np.prod(opts['dim'])))
            out = self.lm(x)
            return out, x

        def get_embedding_dim(self):
            return int(np.prod(opts['dim']))

    # mlp model class
    class mlpMod(nn.Module):
        def __init__(self, dim, embSize=256):
            super(mlpMod, self).__init__()
            self.embSize = embSize
            self.dim = int(np.prod(dim))
            self.lm1 = nn.Linear(self.dim, embSize)
            self.lm2 = nn.Linear(embSize, opts['nClasses'])

        def forward(self, x):
            x = x.view(-1, self.dim)
            emb = F.relu(self.lm1(x))
            out = self.lm2(emb)
            return out, emb

        def get_embedding_dim(self):
            return self.embSize

    # mlp model class
    class mlpDropMod(nn.Module):
        def __init__(self, dim, embSize=256):
            super(mlpDropMod, self).__init__()
            self.embSize = embSize
            self.dim = int(np.prod(dim))
            self.lm1 = nn.Linear(self.dim, embSize)
            self.drop= nn.Dropout(0.5)
            self.lm2 = nn.Linear(embSize, opts['nClasses'])

        def forward(self, x):
            x = x.view(-1, self.dim)
            emb = F.relu(self.lm1(x))
            out = self.lm2(self.drop(emb))
            return out, emb

        def get_embedding_dim(self):
            return self.embSize

    # load specified network
    if opts['model'] == 'mlp' and opts['alg'] == 'bald':
        net = mlpDropMod(opts['dim'], embSize=opts['nEmb'])
        net_test = mlpDropMod(opts['dim'], embSize=opts['nEmb'])
        net_test2 = mlpDropMod(opts['dim'], embSize=opts['nEmb'])
    elif opts['model'] == 'mlp':
        net = mlpMod(opts['dim'], embSize=opts['nEmb'])
        net_test = mlpMod(opts['dim'], embSize=opts['nEmb'])
        net_test2 = mlpMod(opts['dim'], embSize=opts['nEmb'])
    elif opts['model'] == 'resnet':
        net = resnet.ResNet18()
        net_test = resnet.ResNet18()
        net_test2 = resnet.ResNet18()
    elif opts['model'] == 'vgg' and opts['alg'] == 'bald':
        net = vgg.VGGDropout('VGG16', opts['nClasses'])
        net_test = vgg.VGGDropout('VGG16', opts['nClasses'])
        net_test2 = vgg.VGGDropout('VGG16', opts['nClasses'])
    elif opts['model'] == 'vgg':
        net = vgg.VGG('VGG16', opts['nClasses'])
        net_test = vgg.VGG('VGG16', opts['nClasses'])
        net_test2 = vgg.VGG('VGG16', opts['nClasses'])
    else:
        print('choose a valid model - mlp, resnet, or vgg', flush=True)
        raise ValueError

    if opts['did'] > 0 and opts['model'] != 'mlp':
        print('openML datasets only work with mlp', flush=True)
        raise ValueError

    if type(X_tr[0]) is not np.ndarray:
        X_tr = X_tr.numpy()

    filter_factor = opts['filter']
    startpl = opts['start']
    startparam = opts['startparam']
    stepstart = startpl == 'step'
    overlapstart = startpl == 'overlap'
    accuracystart = startpl == 'accuracy'
    entropystart = startpl == 'entropy'

    noearlystop = opts['noearlystop']

    epochnum = opts['epochnum']

    veripl = opts['veri']
    plratio = opts['plratio']

    filtertype = opts['filtertype']

    # set up the specified sampler
    if opts['alg'] == 'badge':
        if opts['plmethod'] == 'none':
            strategy = BadgeSamplingPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
            startparam = NUM_ROUND + 2
            stepstart = True
            overlapstart = False
            accuracystart = False
            entropystart = False
        elif opts['plmethod'] == 'consistency':
            strategy = BadgeSamplingPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'gradient':
            strategy = BadgeSamplingGradientPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'gradient2':
            strategy = BadgeSamplingGradientPL2(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'gradient3':
            strategy = BadgeSamplingGradientPL3(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'random':
            strategy = BadgeSamplingRandPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'entropy':
            strategy = BadgeSamplingEntropyPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'margin':
            strategy = BadgeSamplingMarginPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        else:
            print('choose a valid pseudo labeling function', flush=True)
            raise ValueError
    elif opts['alg'] == 'bald':
        if opts['plmethod'] == 'none':
            strategy = BALDDropout(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
            startparam = NUM_ROUND + 2
            stepstart = True
            overlapstart = False
            accuracystart = False
            entropystart = False
        elif opts['plmethod'] == 'consistency':
            strategy = BALDDropoutConsistencyPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'gradient':
            strategy = BALDDropoutGradientPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'gradient3':
            strategy = BALDDropoutPL3(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'random':
            strategy = BALDDropoutRandPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'entropy':
            strategy = BALDDropoutEntropyPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'margin':
            strategy = BALDDropoutMarginPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'bald':
            strategy = BALDDropoutBALDInv(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'baldinv':
            strategy = BALDDropoutBALDInv(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        else:
            print('choose a valid pseudo labeling function', flush=True)
            raise ValueError
    elif opts['alg'] == 'margin':
        if opts['plmethod'] == 'none':
            strategy = Margin2(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
            startparam = NUM_ROUND + 2
            stepstart = True
            overlapstart = False
            accuracystart = False
            entropystart = False
        else:
            strategy = Margin2(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
    elif opts['alg'] == 'entropy':
        if opts['plmethod'] == 'none':
            strategy = Entropy2(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
            startparam = NUM_ROUND + 2
            stepstart = True
            overlapstart = False
            accuracystart = False
            entropystart = False
        else:
            strategy = Entropy2(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
    else:
        print('choose a valid active  function', flush=True)
        raise ValueError

    # print info
    if opts['did'] > 0: DATA_NAME = 'OML' + str(opts['did'])
    print(DATA_NAME, flush=True)
    print(type(strategy).__name__, flush=True)

    # round 0 accuracy
    acctr = np.zeros(NUM_ROUND + 1)
    acctr[0] = strategy.train(noearlystop, epochnum)

    startstep = time.time()
    P = strategy.predict(X_te, Y_te)
    acc = np.zeros(NUM_ROUND + 1)
    acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)

    print("Main: " + str(NUM_INIT_LB) + '\ttesting accuracy {}'.format(acc[0]), flush=True)

    acctest = np.zeros(NUM_ROUND + 1)
    overlap = np.zeros(NUM_ROUND + 1)
    entropy = np.zeros(NUM_ROUND + 1)
    plpred = np.zeros(NUM_ROUND + 1)

    # case with pseudolabels
    pred_pl = np.zeros(n_pool)

    if opts['veri'] == 3:
        verifier = PLVerifier(X_tr, Y_tr, idxs_lb, net_test, net_test2, handler, n_pool, args, [], pred_pl,
                              filter_factor)
    else:
        if opts['alg'] == 'bald':
            verifier = BALDVerifier(X_tr, Y_tr, idxs_lb, net_test, handler, n_pool, args, [], pred_pl)
        else:
            verifier = BasicVerifier(X_tr, Y_tr, idxs_lb, net_test, handler, n_pool, args, [], pred_pl)

    pl_idxs_store = []
    idxs_pl_known = np.zeros(n_pool, dtype=bool)

    started = False
    # print(strategy.predict_prob(X_tr, Y_tr))
    entropy[0] = calc_entropy(strategy.predict_prob(X_tr, Y_tr).numpy())

    #print("margin:" + str(calc_margin(strategy.predict_prob(X_tr, Y_tr).numpy())))

    plpred[0] = 1

    if (opts['loss'] == 'overlap'):
        P_tr = strategy.predict(X_tr, Y_tr)
        verifier.updatePL(idxs_lb, idxs_pl, pred_pl, pl_idxs_store)
        verifier.train(noearlystop, epochnum)
        P_test_tr = verifier.predict(X_tr, Y_tr)
        overlap[0] = 1.0 * (P_tr == P_test_tr).sum().item() / len(P_tr)
        print('Overlap {}'.format(overlap[0]), flush=True)

    result = {f"acctr": acctr[0], f"acc": acc[0], f"acc_veri": acctest[0], f"pl": 0, f"entropy": entropy[0],
              f"lbpl": NUM_INIT_LB, f"overlap": overlap[0] , f"plcor_all": 0,
              f"plcor_new": 0, f"entropypl": 0, f"plpred": plpred[0]}
    print(str(result))

    endstep = time.time()
    print("Time: "+ str(endstep - startstep))
    startstep = endstep

    print("ClassesLBTrue: " + str(strategy.getDistributionTrueLB()))
    print("ClassesLBPredict: " + str(strategy.getDistributionPredictLB()))
    print("ClassesLBNew: " + str(strategy.getDistributionTrueLB()))

#    print("0: " + str(Y_tr[0]))
#    print("1: " + str(Y_tr[1]))

    numPL = NUM_PSEUDO


    for rd in range(1, NUM_ROUND + 1):
        print('Round {}'.format(rd), flush=True)

        # verifier network handling for early verification
        if (veripl >= 4):
            # verifier.update(idxs_lb)
            verifier.updatePL(idxs_lb, idxs_pl, pred_pl, pl_idxs_store)
            verifier.train(noearlystop, epochnum)

        # query samples for oracle
        output = strategy.query(NUM_QUERY)
        q_idxs = output
        idxs_lb[q_idxs] = True

        # report weighted accuracy
        corr = (strategy.predict(X_tr[q_idxs], torch.Tensor(Y_tr.numpy()[q_idxs]).long())).numpy() == \
               Y_tr.numpy()[q_idxs]

        corr_mean = np.mean(corr)

        # update with removed oracled pseudo-labels + new labeled samples
        strategy.update(idxs_lb)

        if not started:
            started = (rd > startparam or not stepstart)
            started = (overlap[rd - 1] > startparam or not overlapstart) and started
            started = (acc[rd - 1] > startparam or not accuracystart) and started
            started = (entropy[rd - 1] < startparam or not entropystart) and started

        # compute pseudo labels

        # pseudo-label dropping handling
        if opts["remove"]:
            numPL = rd * NUM_PSEUDO
            pl_idxs_store = []

        # replenishment handling
        if opts['replenish']:
            numPL = rd * NUM_PSEUDO - sum(idxs_pl)


        # pseudolabel selection
        if started:

            if opts['dual']:
                strategy.trainPL(plweight, plratio, noearlystop, epochnum)
            # get pseudo labeled samples
            pl_idxs = strategy.queryPL(numPL)
            for pl_idx in pl_idxs:
                pl_idxs_store.append(pl_idx)
        else:
            pl_idxs = []

        # filter update
        strategy.updatefilter(filter_factor - rd*opts['filterdecay'])

        repls = idxs_pl_known[pl_idxs].sum().item()
        idxs_pl[pl_idxs] = True
        idxs_pl_known[pl_idxs] = True

        # remove oracled pseudo labels
        for q_idx in q_idxs:
            if q_idx in pl_idxs_store: pl_idxs_store.remove(q_idx)
            idxs_pl[q_idx] = False

        if (opts['meta'] == 0):
            pred_pl[pl_idxs] = (strategy.predict(X_tr[pl_idxs], torch.Tensor(Y_tr.numpy()[pl_idxs]).long())).numpy()

        # secondary verification network training
        if ((veripl > 0 and veripl < 4) or opts['meta'] != 0 or opts['loss'] == 'overlap'):
            # verifier.update(idxs_lb)
            verifier.updatePL(idxs_lb, idxs_pl, pred_pl, pl_idxs_store)
            verifier.train(noearlystop, epochnum)

        # compute pseudo label labels
        if (opts['meta'] != 0):
            pred_pl[pl_idxs] = (verifier.predict(X_tr[pl_idxs], torch.Tensor(Y_tr.numpy()[pl_idxs]).long())).numpy()

        # check pseudo label correctness
        correctPL = 1.0 * (Y_tr[pl_idxs].numpy() == pred_pl[pl_idxs]).sum().item()
        ratioPL = 0
        if len(pl_idxs) > 0:
            ratioPL = correctPL / len(pl_idxs)

        correctness = dict(zip(pl_idxs, Y_tr[pl_idxs].numpy() == pred_pl[pl_idxs]))
        if (veripl > 0):
            verification = dict(zip(pl_idxs_store, verifier.verify(pl_idxs_store, pred_pl)))

        # append to storage for later checks
        for key in pl_idxs_store:
            if key in verification_store:
                if (veripl > 0):
                    verification_store[key].append(verification[key])
                else:
                    verification_store[key].append(True)
            else:
                correctness_store[key] = correctness[key]
                if (veripl > 0):
                    verification_store[key] = [verification[key]]
                else:
                    verification_store[key] = [True]

        # verification handling
        ctrcorver = 0
        ctrcornver = 0
        ctrncorver = 0
        ctrncornver = 0
        ctrcor = 0
        ctrncor = 0

        pl_idxs_iter = pl_idxs_store.copy()
        if (veripl > 0):
            for key in pl_idxs_iter:
                if (correctness_store[key] and verification_store[key][-1]):
                    ctrcorver += 1
                elif (correctness_store[key] and not verification_store[key][-1]):
                    ctrcornver += 1
                elif (not correctness_store[key] and verification_store[key][-1]):
                    ctrncorver += 1
                else:
                    ctrncornver += 1

                if correctness_store[key]:
                    ctrcor += 1

                else:
                    ctrncor += 1

                if (len(verification_store[key]) >= opts['veriparam']) and (veripl == 2 or veripl == 3 or veripl == 4):
                    verification_rem = True
                    for veriind in range(1, opts['veriparam'] + 1, 1):
                        verification_rem = verification_rem and (not verification_store[key][-veriind])
                else:
                    verification_rem = False

                verification_perf = (not correctness_store[key]) and veripl == 1
                if verification_rem or verification_perf:
                    idxs_pl[key] = False
                    if key in pl_idxs_store: pl_idxs_store.remove(key)
                    if key in correctness_store: del correctness_store[key]
                    if key in verification_store: del verification_store[key]

        if sum(idxs_pl) > 0:
            pl_cor_all = (Y_tr[pl_idxs_store].numpy() == pred_pl[pl_idxs_store]).sum().item() / len(Y_tr[pl_idxs_store].numpy())
        else:
            pl_cor_all = 0

        print(
            "Number of new pseudo labels " + str(len(pl_idxs)) + "; Number of correct new pseudo labels " + str(
                int(correctPL)) + "; Ratio of correct new pseudo labels " + str(
                ratioPL))
        print("RepeatPLs: " + str(repls))

        print("correct + verified: " + str(ctrcorver) + "; correct + not verified " + str(
            ctrcornver) + "; not correct + verified " + str(ctrncorver) + "; not correct + not verified " + str(
            ctrncornver))

        # update
        idxs_pl_new = np.zeros(n_pool, dtype=bool)
        idxs_pl_new[pl_idxs] = idxs_pl[pl_idxs]

        idxs_pl_incor = np.zeros(n_pool, dtype=bool)
        idxs_pl_incor[Y_tr.numpy() != pred_pl] = idxs_pl[Y_tr.numpy() != pred_pl]


        idxs_lb_new = np.zeros(n_pool, dtype=bool)
        idxs_lb_new[q_idxs] = True

        inconsistencypl, entropyplnew = strategy.testNewPL(idxs_pl_new)
        print("EntropyPLNew: " + str(np.mean(np.array(list(np.nan_to_num(entropyplnew.values()))))))

        strategy.updatepl(idxs_lb, idxs_pl, pred_pl)
        inconsistencypl, entropypl = strategy.testPL()

        entropyplall = np.mean(np.array(list(np.nan_to_num(entropypl.values()))))
        print("EntropyPLAll: " + str(entropyplall))

        # select pseudo-label weight
        plweight = 1
        if (opts['loss'] == 'fixed'):
            plweight = opts['lossparam']
        elif (opts['loss'] == 'timestep'):
            #diff = opts['lossparam'][2] - opts['lossparam'][1]
            plweight = opts['lossparam'][0] * (rd - opts['lossparam'][1]) #/ diff
            plweight = max(plweight, opts['lossmin'])
            plweight = min(plweight, opts['lossmax'])
        elif (opts['loss'] == 'entropy'):
            if(entropyplall) > 0:
                plweight = opts['lossparam'] / (entropyplall / 0.01)
            else:
                plweight = opts['lossmax']
            plweight = max(plweight, opts['lossmin'])
            plweight = min(plweight, opts['lossmax'])
        elif (opts['loss'] == 'entropynew'):
            if(entropyplnew) > 0:
                plweight = opts['lossparam'] / (entropyplnew / 0.01)
            else:
                plweight = opts['lossmax']
            plweight = max(plweight, opts['lossmin'])
            plweight = min(plweight, opts['lossmax'])
        elif (opts['loss'] == 'overlap'):
            plweight = opts['lossparam'] * overlap[rd - 1]
            plweight = max(plweight, opts['lossmin'])
            plweight = min(plweight, opts['lossmax'])
        elif (opts['loss'] == 'accuracy'):
            plweight = opts['lossparam'] * corr_mean
            plweight = max(plweight, opts['lossmin'])
            plweight = min(plweight, opts['lossmax'])

        print("PLWeight: " + str(plweight))


        # train learner network
        acctr[rd] = strategy.trainPL(plweight, plratio, noearlystop, epochnum)
        entropy[rd] = calc_entropy(strategy.predict_prob(X_tr, Y_tr).numpy())

        # log class distributions

        print("ClassesPLTrue: " + str(strategy.getDistributionTruePL()))
        print("ClassesPLPL: " + str(strategy.getDistributionPLPL()))
        print("ClassesPLPredict: " + str(strategy.getDistributionPredictPL()))
        print("ClassesPLNew: " + str(strategy.getDistributionTrue(idxs_pl_new)))
        print("ClassesPLPLNew: " + str(strategy.getDistributionPL(idxs_pl_new)))

        print("ClassesPLIncorrect: " + str(strategy.getDistributionTrue(idxs_pl_incor)))

        print("ClassesPLPLPredict: " + str(strategy.getDistributionPLPLPredicted()))
        plpred[rd] = strategy.getPLPred()


        print("ClassesLBTrue: " + str(strategy.getDistributionTrueLB()))
        print("ClassesLBPredict: " + str(strategy.getDistributionPredictLB()))
        print("ClassesLBNew: " + str(strategy.getDistributionTrue(idxs_lb_new)))

        # round accuracy
        P = strategy.predict(X_te, Y_te)
        acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
        print("Main: " + str(sum(idxs_lb) + sum(idxs_pl)) + '\t' + 'testing accuracy {}'.format(acc[rd]), flush=True)
        if (veripl > 0 or opts['loss'] == 'overlap' or overlapstart):
            P_test = verifier.predict(X_te, Y_te)
            acctest[rd] = 1.0 * (Y_te == P_test).sum().item() / len(Y_te)
            print("Test: " + str(sum(idxs_lb)) + '\t' + 'testing accuracy {}'.format(acctest[rd]), flush=True)

            P_tr = strategy.predict(X_tr, Y_tr)
            P_test_tr = verifier.predict(X_tr, Y_tr)
            overlap[rd] = 1.0 * (P_tr == P_test_tr).sum().item() / len(P_tr)
            print('Overlap {}'.format(overlap[rd]), flush=True)
        else:
            acctest[rd] = 0
            overlap[rd] = 0

        #print("margin:" + str(calc_margin(strategy.predict_prob(X_tr, Y_tr).numpy())))

        result = {f"acctr": acctr[rd], f"acc": acc[rd], f"acc_veri": acctest[rd], f"pl": sum(idxs_pl), f"entropy": entropy[rd],
                  f"lbpl": (sum(idxs_lb) + sum(idxs_pl)), f"overlap": overlap[rd], f"plcor_all": pl_cor_all,
                  f"plcor_new": ratioPL, f"entropypl": entropyplall, f"plpred": plpred[rd]}
        print(str(result))

        endstep = time.time()
        print("Time: " + str(endstep - startstep))
        startstep = endstep

        if opts["remove"]:
            idxs_pl = np.zeros(n_pool, dtype=bool)

            strategy.updatepl(idxs_lb, idxs_pl, pred_pl)

        # stop condition with accuracy target
        #if acc[rd] > 0.95:
        #    # if(rd >= 10):
        #    print("Goal reached")
            #database.finalise_experiment()
        #    break
        if sum(~strategy.idxs_lb) < opts['nQuery']:
            #database.finalise_experiment()
            sys.exit('too few remaining points to query')

    #database.finalise_experiment()

#Usage:
# experiment name (fixed for each base/AL method) + Variant (base, ceal, verips, fverips) + taskdata set

if __name__ == '__main__':
    taskid = str(sys.argv[1])
    taskvar = str(sys.argv[2])
    taskds = str(sys.argv[3])

    seed = int(sys.argv[4])



    print(torch.cuda.memory_summary(device=None, abbreviated=False))


    print("Id: " + str(taskid))

    task = opts.copy()

    if (taskds == "CIFAR10s"):
        task.update({'data': 'CIFAR10', 'lr': 0.001, 'epochnum': 100, 'nStart': 100, 'nQuery': 1000, 'nEnd': 10100})
    elif (taskds == "SVHN"):
        task.update({'data': 'SVHN', 'lr': 0.001, 'epochnum': 200, 'nStart': 100, 'nQuery': 1000, 'nEnd': 5100})
    elif (taskds == "MNIST"):
        task.update({'data': 'MNIST', 'model': 'mlp', 'lr': 0.001, 'nStart': 20, 'epochnum': 100, 'nQuery': 100, 'nEnd': 5020})
    elif (taskds == "CIFAR10s"):
        task.update({'data': 'CIFAR10s', 'lr': 0.001, 'epochnum': 100, 'nQuery': 100, 'nEnd': 10100})

    if(taskvar == "base"):
        task.update({'startparam': 100})
    elif(taskvar == "ceal"):
        task.update({'remove': True, 'nPseudo': 1000000})
    elif (taskvar == "verips"):
        task.update({'veri': 2, 'nPseudo': 1000000, 'replenish': True})
    elif (taskvar == "fverips"):
        task.update({'veri': 4, 'nPseudo': 1000000, 'replenish': True})

    if(taskid == "bald"):
        task.update({'alg': 'bald', 'plmethod': 'bald', 'filtertype': 'bald', 'filter': -0.001})
    elif (taskid == "badge"):
        task.update({'alg': 'badge', 'plmethod': 'gradient3', 'filtertype': 'badge', 'filter': 0.0025, 'filterdecay': 0.000033})
    elif (taskid == "entropy"):
        task.update({'alg': 'entropy', 'plmethod': 'entropy', 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    elif(taskid == "margin"):
        task.update({'alg': 'margin', 'plmethod': 'margin', 'filtertype': 'margintrue', 'filter': 0.95})
    elif (taskid == "entropyeq50"):
        task.update({'alg': 'entropy', 'plmethod': 'entropy', 'nStart': 50, 'startdata': 'equal', 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    elif (taskid == "entropyoneout"):
        task.update({'alg': 'entropy', 'plmethod': 'entropy', 'startdata': 'oneout', 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    elif (taskid == "entropyimb"):
        task.update({'alg': 'entropy', 'plmethod': 'entropy', 'startdata': 'imbalance', 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    elif (taskid == "entropy0.1"):
        task.update({'alg': 'entropy', 'plmethod': 'entropy', 'filtertype': 'entropy', 'filter': 0.1, 'filterdecay': 0.0033})
    elif (taskid == "entropy0.06"):
        task.update({'alg': 'entropy', 'plmethod': 'entropy', 'filtertype': 'entropy', 'filter': 0.06, 'filterdecay': 0.0033})
    elif (taskid == "entropy0.04"):
        task.update({'alg': 'entropy', 'plmethod': 'entropy', 'filtertype': 'entropy', 'filter': 0.04, 'filterdecay': 0.0033})
    elif (taskid == "entropydc0.004"):
        task.update({'alg': 'entropy', 'plmethod': 'entropy', 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.004})
    elif (taskid == "entropydc0.002"):
        task.update({'alg': 'entropy', 'plmethod': 'entropy', 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.002})
    elif (taskid == "entropydc0"):
        task.update({'alg': 'entropy', 'plmethod': 'entropy', 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0})

    print(task)

    os.makedirs("experiment_logs", exist_ok=True)
    log_str = "experiment_logs/" + "Results_" + str(task['data']) + "_" + str(task['nStart'])
    log_str += "_q" + str(task['nQuery']) + "_p" + str(task['nPseudo']) + "_" + str(task['alg'])
    log_str += "_" + str(task['plmethod'])
    log_str += "_" + "f" + str(task['filtertype'])  + str(task['filter']) + "s" + str(task['start']) + "sp" + str(task['startparam'])
    log_str += "_" + "l" + str(task['loss']) + "lp" + str(task['lossparam']) + "lm" + str(task['lossmin'])
    log_str += str(task['lossmax']) + "_" + "v" + str(veristr[task['veri']]) + "vp" + str(task['veriparam'])

    if task['remove']:
        log_str += "_rem"
    if not task['filterdecay'] == 0:
        log_str += "_" + str(task['filterdecay'])
    if task['lr'] == 0.001:
        log_str += "_lr"

    if task['noearlystop']:
        log_str += "_nostop"

    if task['replenish']:
        log_str += "_repl"

    if task['startdata'] == 'oneout':
        log_str += "_oneout"

    if task['startdata'] == 'equal':
        log_str += "_equal"

    if task['startdata'] == 'imbalance':
        log_str += "_imb"

    runname = taskid + taskvar + taskds

    start = time.time()
    set_seed(seed)
    log_file = log_str + "_" + str(seed) + ".txt"
    sys.stdout = open(log_file, 'wt')
    print("Id: " + str(taskid))
    main(log_file, task, log_str + "_" + str(seed), runname + str(seed))
    end = time.time()
    print(end - start)
