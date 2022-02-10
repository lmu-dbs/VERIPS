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
import mlflow_logger
import random
import math
import collections

from query_strategies import RandomSampling, BadgeSampling, \
    BaselineSampling, LeastConfidence, MarginSampling, \
    EntropySampling, CoreSet, ActiveLearningByLearning, \
    LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
    KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
    AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning, \
    ConsistencyBased, BadgeSamplingPL, BadgeSamplingRandPL, \
    BadgeSamplingEntropyPL, BadgeSamplingMarginPL, \
    BadgeSamplingGradientPL, BadgeSamplingRand, BALDDropoutRand, BALDDropoutRandPL, \
    BALDDropoutEntropyPL, BALDDropoutMarginPL, BALDDropoutGradientPL, BALDDropoutConsistencyPL, \
    ConsistencyBasedRand, ConsistencyBasedEntropyPL, ConsistencyBasedGradientPL, \
    ConsistencyBasedRandPL, ConsistencyBasedMarginPL, BadgeSamplingGradientPL2, BadgeSamplingGradientPL3, \
    BALDDropoutPL3, BALDND, BALDNDRandPL, BALDNDPL3,\
    BALDNDEntropyPL, BALDNDMarginPL, BALDNDGradientPL, BALDNDConsistencyPL,\
    BALDDropoutBALD, Margin2, Entropy2, BALDDropoutBALDInv

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
# remove: use pseudo-label dropping
# plratio: use ratio-based weighting
# noearlystop: don't use a fixed training stopping point (otherwise given in 'epochnum')
# dual: retrain the model prior to pseudo-label selection
# replenish: try to always maintaining the maximal allowed number of pseudo-labels

opts = {'model': 'vgg', 'nQuery': 1000, 'nPseudo': 1000, 'lr': 1e-4, 'data': 'CIFAR10', 'alg': 'badge',
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

    #database = mlflow_logger.MLFlowLogger()
    #run_id, output_path = database.init_experiment(runname, hyper_parameters=opts)

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
# transform from @yangarbiter: https://gist.github.com/yangarbiter/33a706011d1a833485fdc5000df55d25
    args_pool['CAL'] = {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 128, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.001, 'momentum': 0.5}}


    args_pool['CIFAR10']['transform'] = args_pool['CIFAR10']['transformTest']  # remove data augmentation
    args_pool['CIFAR10s']['transform'] = args_pool['CIFAR10s']['transformTest']
    args_pool['MNISTs']['transformTest'] = args_pool['MNISTs']['transform']
    args_pool['MNIST']['transformTest'] = args_pool['MNIST']['transform']
    args_pool['SVHN']['transformTest'] = args_pool['SVHN']['transform']
    args_pool['CAL']['transformTest'] = args_pool['CAL']['transform']

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

    # cal model class
    class calMod(nn.Module):
        def __init__(self, dim, embSize=256):
            super(calMod, self).__init__()
            self.embSize = embSize
            self.dim = int(np.prod(dim))
            #print("Cal dim: " + str(self.dim))
            #print("Cal class:  " + str(opts['nClasses']))
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LocalResponseNorm(2),
                nn.Conv2d(96, 256, kernel_size=5, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LocalResponseNorm(2),
                nn.Conv2d(256, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )

            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, opts['nClasses']),
                nn.Softmax(1)
            )

        def forward(self, x):
            out = self.features(x)
            emb = out.view(out.size(0), -1)
            out = self.classifier(emb)
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
    elif opts['model'] == 'calm':
        net = calMod(opts['dim'], embSize=opts['nEmb'])
        net_test = calMod(opts['dim'], embSize=opts['nEmb'])
        net_test2 = calMod(opts['dim'], embSize=opts['nEmb'])
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
    elif opts['alg'] == 'randsamp':
        strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        stepstart = True
        overlapstart = False
        accuracystart = False
        entropystart = False


    elif opts['alg'] == 'consistency':
        if opts['plmethod'] == 'none':
            strategy = ConsistencyBased(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
            startparam = NUM_ROUND + 2
            stepstart = True
            overlapstart = False
            accuracystart = False
            entropystart = False
        elif opts['plmethod'] == 'consistency':
            strategy = ConsistencyBased(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'gradient':
            strategy = ConsistencyBasedGradientPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'random':
            strategy = ConsistencyBasedRandPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'entropy':
            strategy = ConsistencyBasedEntropyPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'margin':
            strategy = ConsistencyBasedMarginPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
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
            strategy = BALDDropoutBALD(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'baldinv':
            strategy = BALDDropoutBALDInv(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        else:
            print('choose a valid pseudo labeling function', flush=True)
            raise ValueError
    elif opts['alg'] == 'baldnd':
        if opts['plmethod'] == 'none':
            strategy = BALDND(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
            startparam = NUM_ROUND + 2
            stepstart = True
            overlapstart = False
            accuracystart = False
            entropystart = False
        elif opts['plmethod'] == 'consistency':
            strategy = BALDNDConsistencyPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'gradient':
            strategy = BALDNDGradientPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'gradient3':
            strategy = BALDNDPL3(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'random':
            strategy = BALDNDRandPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'entropy':
            strategy = BALDNDEntropyPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        elif opts['plmethod'] == 'margin':
            strategy = BALDNDMarginPL(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
        else:
            print('choose a valid pseudo labeling function', flush=True)
            raise ValueError
    elif opts['alg'] == 'badgerand':
        if opts['plmethod'] == 'none':
            strategy = BadgeSamplingRand(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
            startparam = NUM_ROUND + 2
            stepstart = True
            overlapstart = False
            accuracystart = False
            entropystart = False
        else:
            print('choose a valid pseudo labeling function', flush=True)
            raise ValueError
    elif opts['alg'] == 'consistencyrand':
        if opts['plmethod'] == 'none':
            strategy = ConsistencyBasedRand(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
            startparam = NUM_ROUND + 2
            stepstart = True
            overlapstart = False
            accuracystart = False
            entropystart = False
        else:
            print('choose a valid pseudo labeling function', flush=True)
            raise ValueError
    elif opts['alg'] == 'baldrand':
        if opts['plmethod'] == 'none':
            strategy = BALDDropoutRand(X_tr, Y_tr, idxs_lb, net, handler, args, filter_factor, filtertype)
            startparam = NUM_ROUND + 2
            stepstart = True
            overlapstart = False
            accuracystart = False
            entropystart = False
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

    #    if opts['alg'] == 'rand':  # random sampling
    #        strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
    #    elif opts['alg'] == 'baseline':  # badge but with k-DPP sampling instead of k-means++
    #        strategy = BaselineSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
    #    elif opts['alg'] == 'cons':  # consistency-based learning
    #        strategy = ConsistencyBased(X_tr, Y_tr, idxs_lb, net, handler, args)
    #    elif opts['alg'] == 'badgepl':
    #        strategy = BadgeSamplingPL(X_tr, Y_tr, idxs_lb, net, handler, args)
    #    elif opts['alg'] == 'badgerandpl':
    #        strategy = BadgeSamplingRandPL(X_tr, Y_tr, idxs_lb, net, handler, args)
    #    elif opts['alg'] == 'badgeaugpl':
    #        strategy = BadgeSamplingAugPL(X_tr, Y_tr, idxs_lb, net, handler, args)
    #    elif opts['alg'] == 'badgestrongaugpl':
    #        strategy = BadgeSamplingStrongAugPL(X_tr, Y_tr, idxs_lb, net, handler, args)
    #    elif opts['alg'] == 'badgeentropypl':
    #        strategy = BadgeSamplingEntropyPL(X_tr, Y_tr, idxs_lb, net, handler, args)
    #    elif opts['alg'] == 'badgemarginpl':
    #        strategy = BadgeSamplingMarginPL(X_tr, Y_tr, idxs_lb, net, handler, args)

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

    strategy.logPredictions()
    curRoundStore = runPrefix + "_0.txt"

    endstep = time.time()
    print("Time: "+ str(endstep - startstep))
    startstep = endstep

    strategy.logToFile(curRoundStore)
    #database.log_results(result=result, step=np.sum(idxs_lb), file=log_file, curRunStore=curRoundStore)
    os.remove(curRoundStore)

    print("ClassesLBTrue: " + str(strategy.getDistributionTrueLB()))
    print("ClassesLBPredict: " + str(strategy.getDistributionPredictLB()))
    print("ClassesLBNew: " + str(strategy.getDistributionTrueLB()))

#    print("0: " + str(Y_tr[0]))
#    print("1: " + str(Y_tr[1]))

    numPL = NUM_PSEUDO


    for rd in range(1, NUM_ROUND + 1):
        print('Round {}'.format(rd), flush=True)
        strategy.clear_store()
        curRoundStore = runPrefix + "_" + str(rd) + ".txt"

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

        strategy.logPredictions()
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

        strategy.logToFile(curRoundStore)

        #database.log_results(result=result, step=np.sum(idxs_lb), file=log_file, curRunStore = curRoundStore)
        os.remove(curRoundStore)

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

# Due to the large amount of parameters and repetition on multiple baselines,
# the original parameter input method was replaced with calls on specific sets
# to minimize the number of parameters needed to be correctly specified upon usage.
# This limits the usability in practice, though in practice the individual parts
# should be split to just use the successful variants anyway.
# This code was intended for experiments only, not practical usage.
# It is also intended to be called with run.sh
# Additionally, it still includes several benchmarking outputs that provide metrics during the runtime
# The results are put into a preset folder and added to MLFlow
# The call is as follows (for usage fo run.sh):
# run.sh TaskID ActiveLearningAlgroithm PseudoLabelingMethod FilterID DataSet ActiveLearningNum PseudoLabelNum
# ActiveLearningAlgroithm:
#           badge: "BADGE" (default)
#           bald: "BALD"
#           cons: "CSAL"
# PseudoLabelingMethod:
#           entropy: Entropy-based Pseudo-labeling (default)
#           consistency: Consistency-based Pseudo-labeling
#           gradient: Gradient-based Pseudo-labeling
#           gram: Gram-based Pseudo-labeling
#           margin: Random Pseudo-labeling
#           random: Random Pseudo-labeling
#DataSet:
#           CIFAR10: CIFAR-10 (default)
#           SVHN: SVHN
#FilterID:  x is parameter for filter (presence depenedent on filter)
#           MRx: Margin-ratio-based filter
#           Mx: Margin-based filter
#           Ex: Entropy-based filter
#           EDx: Entropy-based filter with decay
#           False: No filter
# TaskID:
#           0: Baseline Active Learning
#           1: Baseline Active Learning with Pseudo-labels
#           2: Active Learning with Pseudo-labels and Perfect Verification
#           3: Active Learning with Pseudo-labels and Label-based Verification
#           4: Active Learning with Pseudo-labels and Pseudo-Label-based Verification
#           5: Active Learning with Pseudo-labels and Pseudo-label Dropping
#           6: Active Learning with Pseudo-labels and Label-based Verification and Pseudo-label Dropping
#           7: Active Learning with Pseudo-labels and Early Label-based Verification
#           8: Active Learning with Pseudo-labels and Pseudo-Label-based Verification and Pseudo-label Dropping
#           9: Active Learning with Pseudo-labels and Early Label-based Verification and Pseudo-label Dropping
#           10: Active Learning with Pseudo-labels and Fixed Loss 0.5
#           11: Active Learning with Pseudo-labels and Timestep-based Loss 1 (0.1 per step)
#           12: Active Learning with Pseudo-labels and Timestep-based Loss 0.5 (0.05 per step)
#           13: Active Learning with Pseudo-labels and Entropy-based Loss 1
#           14: Active Learning with Pseudo-labels and Entropy-based Loss 0.5
#           15: Active Learning with Pseudo-labels and Entropy-based Loss 2
#           16: Active Learning with Pseudo-labels and Overlap-based Loss 1
#           17: Active Learning with Pseudo-labels and Overlap-based Loss 0.5
#           18: Active Learning with Pseudo-labels and pseudo-labeling start at step 2
#           19: Active Learning with Pseudo-labels and pseudo-labeling start at step 3
#           20: Active Learning with Pseudo-labels and pseudo-labeling start at step 4
#           21: Baseline Active Learning with learning rate 0.001
#           22: Baseline Active Learning with Pseudo-labels with lr 0.001
#           23: Active Learning with Pseudo-labels and Label-based Verification with Replenishment with lr 0.001
#           24: Active Learning with Pseudo-labels and Pseudo-label Dropping with lr 0.001
#           25: Baseline Active Learning with Pseudo-labels with Replenishemnt with lr 0.001
#           26: Active Learning with Pseudo-labels and Label-based Verification and Pseudo-label Dropping with lr 0.001
#           27: Active Learning with Pseudo-labels and Fixed Loss 3
#           28: Active Learning with Pseudo-labels and pseudo-labeling start at step 5
#           29: Active Learning with Pseudo-labels and Label-based Verification with Replenishment
#           30: Active Learning with Pseudo-labels and Early Label-based Verification with Replenishment
#           31: Active Learning with Pseudo-labels and Accuracy-based Loss 1
#           32: Active Learning with Pseudo-labels and Accuracy-based Loss 0.5
#           36: Baseline Active Learning with Pseudo-labels with Replenishemnt
#           38: Active Learning with Pseudo-labels and Fixed Loss 2
#           39: Active Learning with Pseudo-labels and Early Label-based Verification with Replenishment with lr 0.001
#           40: Active Learning with Pseudo-labels and Early Label-based Verification and Pseudo-label Dropping with lr 0.001
#           41: Active Learning with Pseudo-labels and pseudo-labeling start at step 2 with Replenishment
#           42: Active Learning with Pseudo-labels and pseudo-labeling start at step 3 with Replenishment
#           43: Active Learning with Pseudo-labels and pseudo-labeling start at step 4 with Replenishment
#           44: Active Learning with Pseudo-labels and pseudo-labeling start at step 5 with Replenishment
#           45: Active Learning with Pseudo-labels and pseudo-labeling start at step 6 with Replenishment
#           46: Active Learning with Pseudo-labels and pseudo-labeling start at step 7 with Replenishment
#           47: Active Learning with Pseudo-labels and pseudo-labeling start at step 8 with Replenishment
#           48: Active Learning with Pseudo-labels and pseudo-labeling start at step 9 with Replenishment
#           49: Active Learning with Pseudo-labels and pseudo-labeling start at step 10 with Replenishment
#           50: Active Learning with Pseudo-labels and pseudo-labeling start at step 10 and Label-based Verification with Replenishment with lr 0.001
#           51: Active Learning with Pseudo-labels and pseudo-labeling start at step 10 and Label-based Verification with Replenishment
#           52: Active Learning with Pseudo-labels and pseudo-labeling start at step 7 and Label-based Verification with Replenishment with lr 0.001
#           53: Active Learning with Pseudo-labels and pseudo-labeling start at step 10 with Replenishment with lr 0.001


if __name__ == '__main__':
    taskid = sys.argv[1]
    taskal = str(sys.argv[2])
    taskpl = str(sys.argv[3])
    taskfl = str(sys.argv[4])

    i = int(sys.argv[5])
    taskds = "CIFAR10"
    if(len(sys.argv) > 6):
        taskds = str(sys.argv[6])

    numsal = 1000
    if(len(sys.argv) > 7):
        numsal = int(sys.argv[7])
        opts.update({'nQuery': numsal})

    numspl = 1000
    if(len(sys.argv) > 8):
        numspl = int(sys.argv[8])
        opts.update({'nPseudo': numspl})



    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    if (taskfl != "False" and taskfl != "NoLim"):
        opts.update({'plratio': True})

    if (taskfl == "NoLim"):
        opts.update({'noearlystop': True})

    print("Id: " + str(taskid))
    print("AL: " + str(taskal))
    print("PL: " + str(taskpl))
    print("Filter: " + str(taskfl))

# active learning method

    altype = "BADGE "
    if (taskal == "cons"):
        opts.update({'alg': 'consistency'})
        altype = "Consistency "
    elif (taskal == "bald"):
        opts.update({'alg': 'bald'})
        altype = "BALD "
    elif (taskal == "baldnd"):
        opts.update({'alg': 'baldnd'})
        altype = "BALD ND "
    elif (taskal == "margin"):
        opts.update({'alg': 'margin'})
        altype = "Margin "
    elif (taskal == "entropy"):
        opts.update({'alg': 'entropy'})
        altype = "Entropy "
    elif (taskal == "randsamp"):
        opts.update({'alg': 'randsamp'})
        altype = "Random Sampling "

# pseudo-labeling method

    pltype = "EntropyPL "
    if (taskpl == 'consistency'):
        opts.update({'plmethod': 'consistency'})
        pltype = "ConsistencyPL "
    elif (taskpl == 'gram'):
        opts.update({'plmethod': 'gradient'})
        pltype = "GramPL "
    #elif (taskpl == 'gradient2'):
    #    opts.update({'plmethod': 'gradient2'})
    #   pltype = "GradientPL2 "
    elif (taskpl == 'gradient' or taskpl == 'badge' ):
        opts.update({'plmethod': 'gradient3'})
        pltype = "GradientPL "
    elif (taskpl == 'random'):
        opts.update({'plmethod': 'random'})
        pltype = "RandomPL "
    elif (taskpl == 'margin'):
        opts.update({'plmethod': 'margin'})
        pltype = "MarginPL "
    elif (taskpl == 'bald'):
        opts.update({'plmethod': 'baldinv'})
        pltype = "BALDInv "
    elif (taskpl == 'baldinv'):
        opts.update({'plmethod': 'baldinv'})
        pltype = "BALDInv "

# filter types

    if (taskfl == "True" or taskfl == "MR2"):
        opts.update({'filtertype': 'margin', 'filter': 2})
        filtertype = "MargRatioFilter 2 "
    elif (taskfl == "MR15"):
        opts.update({'filtertype': 'margin', 'filter': 1.5})
        filtertype = "MargRatioFilter 1.5 "
    elif (taskfl == "MR25"):
        opts.update({'filtertype': 'margin', 'filter': 2.5})
        filtertype = "MargRatioFilter 2.5 "
    elif (taskfl == "MR3"):
        opts.update({'filtertype': 'margin', 'filter': 3})
        filtertype = "MargRatioFilter 3 "
    elif (taskfl == "MR4"):
        opts.update({'filtertype': 'margin', 'filter': 4})
        filtertype = "MargRatioFilter 4 "
    elif (taskfl == "MR5"):
        opts.update({'filtertype': 'margin', 'filter': 5})
        filtertype = "MargRatioFilter 5 "
    elif (taskfl == "MR10"):
        opts.update({'filtertype': 'margin', 'filter': 10})
        filtertype = "MargRatioFilter 10 "
    elif (taskfl == "MR20"):
        opts.update({'filtertype': 'margin', 'filter': 20})
        filtertype = "MargRatioFilter 20 "
    elif (taskfl == "MR30"):
        opts.update({'filtertype': 'margin', 'filter': 30})
        filtertype = "MargRatioFilter 30 "
    elif (taskfl == "MR50"):
        opts.update({'filtertype': 'margin', 'filter': 50})
        filtertype = "MargRatioFilter 50 "
    elif (taskfl == "MR100"):
        opts.update({'filtertype': 'margin', 'filter': 100})
        filtertype = "MargRatioFilter 100 "
    elif (taskfl == "MR200"):
        opts.update({'filtertype': 'margin', 'filter': 200})
        filtertype = "MargRatioFilter 200 "
    elif (taskfl == "MR500"):
        opts.update({'filtertype': 'margin', 'filter': 500})
        filtertype = "MargRatioFilter 500 "
    elif (taskfl == "MR1000"):
        opts.update({'filtertype': 'margin', 'filter': 1000})
        filtertype = "MargRatioFilter 1000 "
    elif (taskfl == "MR2000"):
        opts.update({'filtertype': 'margin', 'filter': 2000})
        filtertype = "MargRatioFilter 2000 "
    elif(taskfl == "E0001"):
        opts.update({'filtertype': 'entropy', 'filter': 0.001})
        filtertype = "EntropyFilter 0.001 "
    elif(taskfl == "E0005"):
        opts.update({'filtertype': 'entropy', 'filter': 0.005})
        filtertype = "EntropyFilter 0.005 "
    elif(taskfl == "E005"):
        opts.update({'filtertype': 'entropy', 'filter': 0.05})
        filtertype = "EntropyFilter 0.05 "
    elif(taskfl == "E001"):
        opts.update({'filtertype': 'entropy', 'filter': 0.01})
        filtertype = "EntropyFilter 0.01 "
    elif(taskfl == "E01"):
        opts.update({'filtertype': 'entropy', 'filter': 0.1})
        filtertype = "EntropyFilter 0.1 "
    elif(taskfl == "E025"):
        opts.update({'filtertype': 'entropy', 'filter': 0.25})
        filtertype = "EntropyFilter 0.25 "
    elif(taskfl == "E05"):
        opts.update({'filtertype': 'entropy', 'filter': 0.5})
        filtertype = "EntropyFilter 0.5 "
    elif(taskfl == "E1"):
        opts.update({'filtertype': 'entropy', 'filter': 1})
        filtertype = "EntropyFilter 1 "
    elif(taskfl == "ED003"):
        opts.update({'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.003})
        filtertype = "EntropyDecayFilter 0.05 0.003 "
    elif(taskfl == "ED004"):
        opts.update({'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.004})
        filtertype = "EntropyDecayFilter 0.05 0.005 "
    elif(taskfl == "ED002"):
        opts.update({'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.002})
        filtertype = "EntropyDecayFilter 0.05 0.01 "
    elif (taskfl == "M9"):
        opts.update({'filtertype': 'margintrue', 'filter': 0.9})
        filtertype = "MarginFilter 0.9 "
    elif (taskfl == "M95"):
        opts.update({'filtertype': 'margintrue', 'filter': 0.95})
        filtertype = "MarginFilter 0.95 "
    elif (taskfl == "M99"):
        opts.update({'filtertype': 'margintrue', 'filter': 0.99})
        filtertype = "MarginFilter 0.99 "
    elif (taskfl == "M995"):
        opts.update({'filtertype': 'margintrue', 'filter': 0.995})
        filtertype = "MarginFilter 0.995 "
    elif (taskfl == "M999"):
        opts.update({'filtertype': 'margintrue', 'filter': 0.999})
        filtertype = "MarginFilter 0.999 "
    elif (taskfl == "M9995"):
        opts.update({'filtertype': 'margintrue', 'filter': 0.9995})
        filtertype = "MarginFilter 0.9995 "
    elif (taskfl == "M9999"):
        opts.update({'filtertype': 'margintrue', 'filter': 0.9999})
        filtertype = "MarginFilter 0.9999 "
    elif (taskfl == "M8"):
        opts.update({'filtertype': 'margintrue', 'filter': 0.8})
        filtertype = "MarginFilter 0.8 "
    elif (taskfl == "M7"):
        opts.update({'filtertype': 'margintrue', 'filter': 0.7})
        filtertype = "MarginFilter 0.7 "
    elif (taskfl == "M6"):
        opts.update({'filtertype': 'margintrue', 'filter': 0.6})
        filtertype = "MarginFilter 0.6 "
    elif (taskfl == "M5"):
        opts.update({'filtertype': 'margintrue', 'filter': 0.5})
        filtertype = "MarginFilter 0.5 "
    elif (taskfl == "M4"):
        opts.update({'filtertype': 'margintrue', 'filter': 0.4})
        filtertype = "MarginFilter 0.4 "
    elif (taskfl == "M3"):
        opts.update({'filtertype': 'margintrue', 'filter': 0.3})
        filtertype = "MarginFilter 0.3 "
    elif (taskfl == "M2"):
        opts.update({'filtertype': 'margintrue', 'filter': 0.2})
        filtertype = "MarginFilter 0.2 "
    else:
        filtertype = ""

# varaints based on input number

    runname = {}
    params = {}
    params['1'] = opts.copy()
    runname['1'] = altype + pltype + "Baseline " + filtertype
    params['0'] = opts.copy()
    params['0'].update({'startparam': 100})
    runname['0'] = altype + "Baseline " + filtertype

    params['2'] = opts.copy()
    params['2'].update({'veri': 1})
    runname['2'] = altype + pltype + " Perf " + filtertype
    params['3'] = opts.copy()
    params['3'].update({'veri': 2})
    runname['3'] = altype + pltype + " Single 1 " + filtertype

    params['4'] = opts.copy()
    params['4'].update({'veri': 3})
    runname['4'] = altype + pltype + " Dual 1 " + filtertype

    params['5'] = opts.copy()
    params['5'].update({'remove': True})
    runname['5'] = altype + pltype + "PL Drop " + filtertype

    params['6'] = opts.copy()
    params['6'].update({'veri': 2, 'remove': True})
    runname['6'] = altype + pltype + " Single 1 Drop " + filtertype

    params['7'] = opts.copy()
    params['7'].update({'veri': 4})
    runname['7'] = altype + pltype + " SinglePre 1 " + filtertype

    params['8'] = opts.copy()
    params['8'].update({'veri': 3, 'remove': True})
    runname['8'] = altype + pltype + " Dual 1 Drop " + filtertype

    params['9'] = opts.copy()
    params['9'].update({'veri': 4, 'remove': True})
    runname['9'] = altype + pltype + " SinglePre 1 Drop " + filtertype

    params['10'] = opts.copy()
    params['10'].update({'lossparam': 0.5})
    runname['10'] = altype + pltype + " FixedLoss 0.5 " + filtertype
    params['11'] = opts.copy()
    params['11'].update({'loss':'timestep', 'lossparam': [0.1, 0]})
    runname['11'] = altype + pltype + " TimeLoss 1 " + filtertype
    params['12'] = opts.copy()
    params['12'].update({'loss':'timestep', 'lossparam': [0.05, 0]})
    runname['12'] = altype + pltype + " TimeLoss 0.5 " + filtertype
    params['13'] = opts.copy()
    params['13'].update({'loss':'entropy'})
    runname['13'] = altype + pltype + " EntropyLoss 1 " + filtertype
    params['14'] = opts.copy()
    params['14'].update({'loss':'entropy', 'lossparam': 0.5})
    runname['14'] = altype + pltype + " EntropyLoss 0.5 " + filtertype
    params['15'] = opts.copy()
    params['15'].update({'loss':'entropy', 'lossparam': 2})
    runname['15'] = altype + pltype + " EntropyLoss 2 " + filtertype
    params['16'] = opts.copy()
    params['16'].update({'loss':'overlap'})
    runname['16'] = altype + pltype + "PL OverlapLoss 1 " + filtertype
    params['17'] = opts.copy()
    params['17'].update({'loss':'overlap', 'lossparam': 0.5})
    runname['17'] = altype + pltype + "PL OverlapLoss 0.5 " + filtertype

    params['18'] = opts.copy()
    params['18'].update({'startparam': 1})
    runname['18'] = altype + pltype + " Start 1 " + filtertype

    params['19'] = opts.copy()
    params['19'].update({'startparam': 2})
    runname['19'] = altype + pltype + " Start 2 " + filtertype

    params['20'] = opts.copy()
    params['20'].update({'startparam': 3})
    runname['20'] = altype + pltype + " Start 3 " + filtertype

    params['21'] = opts.copy()
    params['21'].update({'startparam': 100, 'lr': 0.001, 'epochnum': 100})
    runname['21'] = altype + "BaselineLR " + filtertype

    params['22'] = opts.copy()
    params['22'].update({'lr': 0.001, 'epochnum': 100})
    runname['22'] = altype + pltype + "BaselineLR " + filtertype

    params['23'] = opts.copy()
    params['23'].update({'veri': 2, 'lr': 0.001, 'epochnum': 100, 'replenish': True})
    runname['23'] = altype + pltype + "BaselineLR Single 1 Replenish " + filtertype

    params['24'] = opts.copy()
    params['24'].update({'remove': True, 'lr': 0.001, 'epochnum': 100})
    runname['24'] = altype + pltype + "BaselineLR Drop " + filtertype

    params['25'] = opts.copy()
    params['25'].update({'lr': 0.001, 'epochnum': 100, 'replenish': True})
    runname['25'] = altype + pltype + "BaselineLR Replenish " + filtertype

    params['26'] = opts.copy()
    params['26'].update({'remove': True, 'veri': 2, 'lr': 0.001, 'epochnum': 100})
    runname['26'] = altype + pltype + "BaselineLR Single 1 Drop " + filtertype

    params['27'] = opts.copy()
    params['27'].update({'lossparam': 3})
    runname['27'] = altype + pltype + " FixedLoss 3 " + filtertype

    params['28'] = opts.copy()
    params['28'].update({'startparam': 5})
    runname['28'] = altype + pltype + " Start 5 " + filtertype

    params['29'] = opts.copy()
    params['29'].update({'veri': 2, 'replenish': True})
    runname['29'] = altype + pltype + " Single 1 Replenish " + filtertype

    params['30'] = opts.copy()
    params['30'].update({'veri': 4, 'replenish': True})
    runname['30'] = altype + pltype + " SinglePre 1 Replenish " + filtertype

    params['31'] = opts.copy()
    params['31'].update({'loss': 'accuracy'})
    runname['31'] = altype + pltype + "PL AccuracyLoss 1 " + filtertype

    params['32'] = opts.copy()
    params['32'].update({'loss': 'accuracy', 'lossparam': 0.5})
    runname['32'] = altype + pltype + "PL AccuracyLoss 0.5 " + filtertype

    params['36'] = opts.copy()
    params['36'].update({'replenish': True})
    runname['36'] = altype + pltype + " Baseline Replenish " + filtertype

    params['38'] = opts.copy()
    params['38'].update({'lossparam': 2})
    runname['38'] = altype + pltype + " FixedLoss 2 " + filtertype

    params['39'] = opts.copy()
    params['39'].update({'veri': 4, 'lr': 0.001, 'epochnum': 100, 'replenish': True})
    runname['39'] = altype + pltype + "BaselineLR SinglePre 1 Replenish " + filtertype

    params['40'] = opts.copy()
    params['40'].update({'remove': True, 'veri': 4, 'lr': 0.001, 'epochnum': 100})
    runname['40'] = altype + pltype + "BaselineLR SinglePre 1 Drop " + filtertype

    params['41'] = opts.copy()
    params['41'].update({'replenish': True, 'startparam': 1})
    runname['41'] = altype + pltype + " Start 1 Replenish " + filtertype

    params['42'] = opts.copy()
    params['42'].update({'replenish': True, 'startparam': 2})
    runname['42'] = altype + pltype + " Start 2 Replenish " + filtertype

    params['43'] = opts.copy()
    params['43'].update({'replenish': True, 'startparam': 3})
    runname['43'] = altype + pltype + " Start 3 Replenish " + filtertype

    params['44'] = opts.copy()
    params['44'].update({'replenish': True, 'startparam': 4})
    runname['44'] = altype + pltype + " Start 4 Replenish " + filtertype

    params['45'] = opts.copy()
    params['45'].update({'replenish': True, 'startparam': 5})
    runname['45'] = altype + pltype + " Start 5 Replenish " + filtertype

    params['46'] = opts.copy()
    params['46'].update({'replenish': True, 'startparam': 6})
    runname['46'] = altype + pltype + " Start 6 Replenish " + filtertype

    params['47'] = opts.copy()
    params['47'].update({'replenish': True, 'startparam': 7})
    runname['47'] = altype + pltype + " Start 7 Replenish " + filtertype

    params['48'] = opts.copy()
    params['48'].update({'replenish': True, 'startparam': 8})
    runname['48'] = altype + pltype + " Start 8 Replenish " + filtertype

    params['49'] = opts.copy()
    params['49'].update({'replenish': True, 'startparam': 9})
    runname['49'] = altype + pltype + " Start 9 Replenish " + filtertype

    params['50'] = opts.copy()
    params['50'].update({'veri': 2, 'replenish': True, 'startparam': 9, 'lr': 0.001, 'epochnum': 100})
    runname['50'] = altype + pltype + " Start 9 Single 1 Replenish LR " + filtertype

    params['51'] = opts.copy()
    params['51'].update({'veri': 2, 'replenish': True, 'startparam': 9})
    runname['51'] = altype + pltype + " Start 9 Single 1 Replenish " + filtertype

    params['52'] = opts.copy()
    params['52'].update({'veri': 2, 'replenish': True, 'startparam': 6})
    runname['52'] = altype + pltype + " Start 6 Single 1 Replenish " + filtertype

    params['53'] = opts.copy()
    params['53'].update({'replenish': True, 'startparam': 9, 'lr': 0.001, 'epochnum': 100})
    runname['53'] = altype + pltype + " Start 9 Replenish LR " + filtertype

    params['54'] = opts.copy()
    params['54'].update({'lr': 0.001, 'epochnum': 200})
    runname['54'] = altype + pltype + "BaselineLR Long " + filtertype

    params['55'] = opts.copy()
    params['55'].update({'veri': 2, 'lr': 0.001, 'epochnum': 200, 'replenish': True})
    runname['55'] = altype + pltype + "BaselineLR Long Single 1 Replenish " + filtertype

    params['56'] = opts.copy()
    params['56'].update({'remove': True, 'lr': 0.001, 'epochnum': 200})
    runname['56'] = altype + pltype + "BaselineLR Long Drop " + filtertype

    params['57'] = opts.copy()
    params['57'].update({'lr': 0.001, 'epochnum': 200, 'replenish': True})
    runname['57'] = altype + pltype + "BaselineLR Long Replenish " + filtertype

    params['58'] = opts.copy()
    params['58'].update({'veri': 4, 'lr': 0.001, 'epochnum': 200, 'replenish': True})
    runname['58'] = altype + pltype + "BaselineLR Long SinglePre 1 Replenish " + filtertype

    params['59'] = opts.copy()
    params['59'].update({'remove': True, 'nPseudo': 1000000, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['59'] = altype + pltype + "Baseline CEAL "

    params['60'] = opts.copy()
    params['60'].update({'veri': 4, 'nPseudo': 1000000, 'replenish': True, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['60'] = altype + pltype + "Baseline CEAL - Early Verification "

    params['61'] = opts.copy()
    params['61'].update({'remove': True, 'lr': 0.001, 'epochnum': 100, 'nPseudo': 900})
    runname['61'] = altype + pltype + "BaselineLR Drop " + filtertype

    params['62'] = opts.copy()
    params['62'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['62'] = altype + pltype + "Baseline CEAL - Verification "

    params['63'] = opts.copy()
    params['63'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['63'] = altype + pltype + "Baseline CEAL "

    params['64'] = opts.copy()
    params['64'].update({'veri': 4, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['64'] = altype + pltype + "Baseline CEAL - Early Verification "

    params['65'] = opts.copy()
    params['65'].update({'veri': 2, 'nPseudo': 100000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['65'] = altype + pltype + "Baseline CEAL - Verification "

    params['66'] = opts.copy()
    params['66'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'margintrue', 'filter': 0.95, 'filterdecay': -0.0033})
    runname['66'] = altype + pltype + "Baseline MarginFilter "

    params['67'] = opts.copy()
    params['67'].update({'veri': 4, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'margintrue', 'filter': 0.95, 'filterdecay': -0.0033})
    runname['67'] = altype + pltype + "Baseline MarginFilter - Early Verification "

    params['68'] = opts.copy()
    params['68'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'margintrue', 'filter': 0.95, 'filterdecay': -0.0033})
    runname['68'] = altype + pltype + "Baseline MarginFilter - Verification "

    params['69'] = opts.copy()
    params['69'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.005, 'filterdecay': 0.00033})
    runname['69'] = altype + pltype + "Baseline CEAL 0.005 "

    params['70'] = opts.copy()
    params['70'].update({'veri': 4, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.005, 'filterdecay': 0.00033})
    runname['70'] = altype + pltype + "Baseline CEAL 0.005 - Early Verification "

    params['71'] = opts.copy()
    params['71'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.005, 'filterdecay': 0.00033})
    runname['71'] = altype + pltype + "Baseline CEAL 0.005 - Verification "

    params['72'] = opts.copy()
    params['72'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'badge', 'filter': 0.0025, 'filterdecay': 0})
    runname['72'] = altype + pltype + "Baseline BADGE Filter 0.0025 "

    params['73'] = opts.copy()
    params['73'].update({'veri': 4, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'badge', 'filter': 0.0025, 'filterdecay': 0})
    runname['73'] = altype + pltype + "Baseline BADGE Filter 0.0025 - Early Verification "

    params['74'] = opts.copy()
    params['74'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'badge', 'filter': 0.0025, 'filterdecay': 0})
    runname['74'] = altype + pltype + "Baseline BADGE Filter 0.0025 - Verification "

    params['75'] = opts.copy()
    params['75'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'bald', 'filter': 0.0000001})
    runname['75'] = altype + pltype + "Baseline BALD Filter 0.0000001 "

    params['76'] = opts.copy()
    params['76'].update({'veri': 4, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'bald', 'filter': 0.0000001})
    runname['76'] = altype + pltype + "Baseline BALD Filter 0.0000001 - Early Verification "

    params['77'] = opts.copy()
    params['77'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'bald', 'filter': 0.0000001})
    runname['77'] = altype + pltype + "Baseline BALD Filter 0.0000001 - Verification "

    params['78'] = opts.copy()
    params['78'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 200, 'lr': 0.001, 'filtertype': 'margintrue', 'filter': 0.95, 'filterdecay': -0.0033})
    runname['78'] = altype + pltype + "Baseline MarginFilter "

    params['79'] = opts.copy()
    params['79'].update({'veri': 4, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 200, 'lr': 0.001, 'filtertype': 'margintrue', 'filter': 0.95, 'filterdecay': -0.0033})
    runname['79'] = altype + pltype + "Baseline MarginFilter - Early Verification "

    params['80'] = opts.copy()
    params['80'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 200, 'lr': 0.001, 'filtertype': 'margintrue', 'filter': 0.95, 'filterdecay': -0.0033})
    runname['80'] = altype + pltype + "Baseline MarginFilter - Verification "

    params['81'] = opts.copy()
    params['81'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'badge', 'filter': 0.005, 'filterdecay': 0.000033})
    runname['81'] = altype + pltype + "Baseline BADGE Filter 0.005 "

    params['82'] = opts.copy()
    params['82'].update({'veri': 4, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'badge', 'filter': 0.005, 'filterdecay': 0.000033})
    runname['82'] = altype + pltype + "Baseline BADGE Filter 0.005 - Early Verification "

    params['83'] = opts.copy()
    params['83'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'badge', 'filter': 0.005, 'filterdecay': 0.000033})
    runname['83'] = altype + pltype + "Baseline BADGE Filter 0.005 - Verification "

    params['86'] = opts.copy()
    params['86'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'bald', 'filter': 0.0000005})
    runname['86'] = altype + pltype + "Baseline BALD Filter 0.0000005 "

    params['85'] = opts.copy()
    params['85'].update({'veri': 4, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'bald', 'filter': 0.0000005})
    runname['85'] = altype + pltype + "Baseline BALD Filter 0.0000005 - Early Verification "

    params['85'] = opts.copy()
    params['85'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'bald', 'filter': 0.0000005})
    runname['85'] = altype + pltype + "Baseline BALD Filter 0.0000005 - Verification "

    params['86'] = opts.copy()
    params['86'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.06, 'filterdecay': 0.0033})
    runname['86'] = altype + pltype + "Baseline CEAL 0.06 "

    params['87'] = opts.copy()
    params['87'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.06, 'filterdecay': 0.0033})
    runname['87'] = altype + pltype + "Baseline CEAL 0.06 - Verification "

    params['88'] = opts.copy()
    params['88'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.04, 'filterdecay': 0.0033})
    runname['88'] = altype + pltype + "Baseline CEAL 0.04 "

    params['89'] = opts.copy()
    params['89'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.04, 'filterdecay': 0.0033})
    runname['89'] = altype + pltype + "Baseline CEAL 0.04 - Verification "

    params['90'] = opts.copy()
    params['90'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.1, 'filterdecay': 0.0033})
    runname['90'] = altype + pltype + "Baseline CEAL 0.1 "

    params['91'] = opts.copy()
    params['91'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.1, 'filterdecay': 0.0033})
    runname['91'] = altype + pltype + "Baseline CEAL 0.1 - Verification "

    params['92'] = opts.copy()
    params['92'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.004})
    runname['92'] = altype + pltype + "Baseline CEAL 0.05 - Decay 0.004"

    params['93'] = opts.copy()
    params['93'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.004})
    runname['93'] = altype + pltype + "Baseline CEAL 0.05 - Decay 0.004 - Verification "

    params['94'] = opts.copy()
    params['94'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.002})
    runname['94'] = altype + pltype + "Baseline CEAL 0.05 - Decay 0.002"

    params['95'] = opts.copy()
    params['95'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.002})
    runname['95'] = altype + pltype + "Baseline CEAL 0.05 - Decay 0.002 - Verification "

    params['96'] = opts.copy()
    params['96'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.001})
    runname['96'] = altype + pltype + "Baseline CEAL 0.05 - Decay 0.001"

    params['97'] = opts.copy()
    params['97'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.001})
    runname['97'] = altype + pltype + "Baseline CEAL 0.05 - Decay 0.001 - Verification "

    params['98'] = opts.copy()
    params['98'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0})
    runname['98'] = altype + pltype + "Baseline CEAL 0.05 - Decay 0"

    params['99'] = opts.copy()
    params['99'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0})
    runname['99'] = altype + pltype + "Baseline CEAL 0.05 - Decay 0 - Verification "

    params['100'] = opts.copy()
    params['100'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'bald', 'filter': -0.001})
    runname['100'] = altype + pltype + "Baseline BALD Filter -0.001 - CEAL"

    params['101'] = opts.copy()
    params['101'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 200, 'lr': 0.001, 'filtertype': 'bald', 'filter': -0.001})
    runname['101'] = altype + pltype + "Baseline BALD Filter -0.001 - Verification "

    params['102'] = opts.copy()
    params['102'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 200, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.1, 'filterdecay': 0.00033})
    runname['102'] = altype + pltype + "Baseline CEAL 0.00033 "

    params['103'] = opts.copy()
    params['103'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.1, 'filterdecay': 0.00033})
    runname['103'] = altype + pltype + "Baseline CEAL 0.00033 - Verification "

    params['104'] = opts.copy()
    params['104'].update({'remove': True, 'nPseudo': 1000000, 'epochnum': 200, 'lr': 0.001, 'filtertype': 'margintrue', 'filter': 0.995, 'filterdecay': -0.00033})
    runname['104'] = altype + pltype + "Baseline MarginFilter 0.995 Decay -0.00033 - CEAL"

    params['105'] = opts.copy()
    params['105'].update({'veri': 2, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 200, 'lr': 0.001, 'filtertype': 'margintrue', 'filter': 0.995, 'filterdecay': -0.00033})
    runname['105'] = altype + pltype + "Baseline MarginFilter 0.995 Decay -0.00033 - Verification "

    params['106'] = opts.copy()
    params['106'].update({'startparam': 100, 'lr': 0.001, 'epochnum': 100, 'nStart': 50, 'startdata': 'equal'})
    runname['106'] = altype + "BaselineLR small"

    params['107'] = opts.copy()
    params['107'].update({'remove': True, 'nPseudo': 1000000, 'nStart': 50, 'startdata': 'equal', 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['107'] = altype + pltype + "Baseline CEAL small"

    params['108'] = opts.copy()
    params['108'].update({'veri': 2, 'nPseudo': 100000, 'nStart': 50, 'startdata': 'equal', 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['108'] = altype + pltype + "Baseline CEAL small - Verification "

    params['109'] = opts.copy()
    params['109'].update({'veri': 4, 'nPseudo': 1000000, 'replenish': True, 'epochnum': 200, 'lr': 0.001, 'filtertype': 'bald', 'filter': -0.001})
    runname['109'] = altype + pltype + "Baseline BALD Filter -0.001 - Early Verification "

    params['110'] = opts.copy()
    params['110'].update({'startparam': 100, 'lr': 0.001, 'epochnum': 100, 'nStart': 100, 'startdata': 'oneout'})
    runname['110'] = altype + "BaselineLR oneout"

    params['111'] = opts.copy()
    params['111'].update({'remove': True, 'nPseudo': 1000000, 'nStart': 100, 'startdata': 'oneout', 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['111'] = altype + pltype + "Baseline CEAL oneout"

    params['112'] = opts.copy()
    params['112'].update({'veri': 2, 'nPseudo': 100000, 'nStart': 100, 'startdata': 'oneout', 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['112'] = altype + pltype + "Baseline CEAL oneout - Verification "

    params['113'] = opts.copy()
    params['113'].update({'startparam': 100, 'lr': 0.001, 'epochnum': 100, 'nStart': 100, 'startdata': 'imbalance'})
    runname['113'] = altype + "BaselineLR imbalanced"

    params['114'] = opts.copy()
    params['114'].update({'remove': True, 'nPseudo': 1000000, 'nStart': 100, 'startdata': 'imbalance', 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['114'] = altype + pltype + "Baseline CEAL imbalanced"

    params['115'] = opts.copy()
    params['115'].update({'veri': 2, 'nPseudo': 100000, 'nStart': 100, 'startdata': 'imbalance', 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['115'] = altype + pltype + "Baseline CEAL imbalanced - Verification "

    params['116'] = opts.copy()
    params['116'].update({'remove': True, 'nPseudo': 1000000, 'nStart': 1000, 'nEnd': 11000, 'startdata': 'oneout', 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['116'] = altype + pltype + "Baseline CEAL oneout 2"

    params['117'] = opts.copy()
    params['117'].update({'veri': 2, 'nPseudo': 100000, 'nStart': 1000, 'nEnd': 11000, 'startdata': 'oneout', 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['117'] = altype + pltype + "Baseline CEAL oneout 2 - Verification "

    params['118'] = opts.copy()
    params['118'].update({'remove': True, 'nPseudo': 1000000, 'nStart': 1000, 'nEnd': 11000, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['118'] = altype + pltype + "Baseline CEAL 2"

    params['119'] = opts.copy()
    params['119'].update({'veri': 2, 'nPseudo': 100000, 'nStart': 1000, 'nEnd': 11000, 'replenish': True, 'epochnum': 100, 'lr': 0.001, 'filtertype': 'entropy', 'filter': 0.05, 'filterdecay': 0.0033})
    runname['119'] = altype + pltype + "Baseline CEAL 2 - Verification "


    task = params[taskid]

    if (taskds == "SVHN"):
        task.update({'data': 'SVHN', 'lr': 0.001, 'epochnum': 200, 'nQuery': 1000, 'nEnd': 5100})

    if (taskds == "MNIST"):
        task.update({'data': 'MNIST', 'model': 'mlp', 'lr': 0.001, 'nStart': 20, 'epochnum': 100, 'nQuery': 100, 'nEnd': 5020})
        if taskid == '100' or taskid == '101':
            task.update({'epochnum': 200})

    if (taskds == "CAL"):
        task.update({'data': 'CAL', 'model': 'vgg', 'lr': 0.01, 'epochnum': 100, 'nStart': 2400, 'nQuery': 1000, 'nEnd': 12400, 'nClasses': 257})

    if (taskds == "CIFAR10s"):
        task.update({'data': 'CIFAR10s'})

    print(task)

    log_str = "/home/stud/jahnp/Masterthesis/experiment_logs/" + "Results_" + str(task['data']) + "_" + str(task['nStart'])
    log_str += "_q" + str(task['nQuery']) + "_p" + str(task['nPseudo']) + "_" + str(task['alg'])
    log_str += "_" + str(task['plmethod'])
    log_str += "_" + "f" + str(task['filtertype'])  + str(task['filter']) + "s" + str(task['start']) + "sp" + str(task['startparam'])
    log_str += "_" + "l" + str(task['loss']) + "lp" + str(task['lossparam']) + "lm" + str(task['lossmin'])
    log_str += str(task['lossmax']) + "_" + "v" + str(veristr[task['veri']]) + "vp" + str(task['veriparam'])

    if task['remove']:
        log_str += "_rem"
    if task['filterdecay'] > 0:
        log_str += "_" + str(task['filterdecay'])
    if task['lr'] == 0.001:
        log_str += "_lr"
    print(log_str)

    if (taskfl == "Ratio"):
        log_str += "_test"

    if task['noearlystop']:
        log_str += "_nostop"

    if task['replenish']:
        log_str += "_repl"


    if taskid == '54' or taskid == '55' or taskid == '56' or taskid == '57' or taskid == '58' or taskid == '65' or taskid == '63' or taskid == '64' or taskid == '78' or taskid == '79' or taskid == '80':
        log_str += "_long"


    if taskid == '106' or taskid == '107' or taskid == '108':
        log_str += "_small"

    if taskid == '110' or taskid == '111' or taskid == '112':
        log_str += "_oneout"

    if taskid == '113' or taskid == '114' or taskid == '115':
        log_str += "_imb"

    if (taskid == '100' or taskid == '101') and taskds == "MNIST":
        log_str += "_long"

    #log_str += "_time"

    start = time.time()
    num = i + 1
    set_seed(i)
    log_file = log_str + "_" + str(num) + ".txt"
    sys.stdout = open(log_file, 'wt')
    print("Id: " + str(taskid))
    main(log_file, task, log_str + "_" + str(num), runname[taskid] + str(num))
    end = time.time()
    print(end - start)
