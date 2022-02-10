import numpy as np
from torch import nn
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
from torchvision import transforms
import pandas as pd
import collections
import pdb

# data augmentation
def augment(x):
    return nn.Sequential(transforms.RandomAffine(20, translate=(0.25, 0.25), scale=(0.75, 1.25)), )(x)

# basic strategy
class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype):
        pd.set_option('mode.chained_assignment', None)
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        use_cuda = torch.cuda.is_available()
        self.filter_factor = filter_factor
        self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
        self.y_pl = np.zeros(self.n_pool)
        self.filtertype = filtertype

        temp = np.zeros((len(self.Y), 6))
        self.store = pd.DataFrame(temp, columns=['type', 'label', 'pseFudolabel', 'prediction', 'scoring', 'entropy'])
        self.store['pseudolabel'] = -1
        self.store['prediction'] = -1
        self.store['label'] = self.Y
        self.store['type'] = 'ul'

# clear value storage for next step
    def clear_store(self):
        temp = np.zeros((len(self.Y), 6))
        self.store = pd.DataFrame(temp, columns=['type', 'label', 'pseudolabel', 'prediction', 'scoring', 'entropy'])
        self.store['pseudolabel'] = -1
        self.store['prediction'] = -1
        self.store['label'] = self.Y
        self.store['type'] = 'ul'

# query active learning target
    def query(self, n):
        pass

# query pseudo-labels
    def queryPL(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb
        idxs = np.arange(self.n_pool)[self.idxs_lb]
        self.store['type'][idxs] = 'lb'

# train on data
    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        accFinal = 0.
        switch = False
        for batch_idx, (x, y, id, idxs) in enumerate(loader_tr):

            if (len(x) == 1):
                switch = True
                self.clf.eval()

            x, y = Variable(x.cuda()), Variable(y.cuda())
            optimizer.zero_grad()
            out, e1 = self.clf(x)

            #print(y.dtype)
            #print(np.unique(y.cpu().numpy()))
            #print(len(np.unique(y.cpu().numpy())))

            loss = F.cross_entropy(out, y)
            #accFinal += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()
            loss.backward()

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()

            if(switch):
                self.clf.train()
                switch = False

        self.clf.eval()
        for batch_idx, (x, y, id, idxs) in enumerate(loader_tr):
            x, y = Variable(x.cuda()), Variable(y.cuda())
            out, e1 = self.clf(x)
            accFinal += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()
        return accFinal / len(loader_tr.dataset.X)


# 1 for lb, 2 for pl, 3 for ul
# all training cycles
    def train(self, noearlystop, epochnum):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        n_epoch = self.args['n_epoch']
        self.clf = self.net.apply(weight_reset).cuda()
        optimizer = optim.Adam(self.clf.parameters(), lr=self.args['lr'], weight_decay=0)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        id = np.empty(len(idxs_train))
        id[:] = 1

        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(), id,
                                            transform=self.args['transform']), shuffle=True,
                               **self.args['loader_tr_args'])

        epoch = 0
        accCurrent = 0.
        while accCurrent < 0.99 and (epoch < epochnum or noearlystop):
            accCurrent = self._train(epoch, loader_tr, optimizer)
            epoch += 1
            #if (epoch % 50 == 0) and (accCurrent < 0.2):  # reset if not converging
            #    self.clf = self.net.apply(weight_reset)
            #    optimizer = optim.Adam(self.clf.parameters(), lr=self.args['lr'], weight_decay=0)
        print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)
        return accCurrent

# train on data with pseudo-labels
    def _trainPL(self, epoch, loader_tr, optimizer, pl_factor, ratiopl):
        self.clf.train()
        accFinal = 0.
        clb = 0
        cpl = 0

        switch = False

        for batch_idx, (x, y, id, idxs) in enumerate(loader_tr):

            x_lb = x[id == 1]
            y_lb = y[id == 1]
            x_pl = x[id == 2]
            y_pl = y[id == 2]

            clb += len(x_lb)
            cpl += len(x_pl)

            sum = len(x_pl) + len(x_lb)
            #ratiopl = len(x_pl)/sum
            #ratiolb = len(x_lb) / sum

            if (len(x_lb) > 1):
                if(len(x_lb) == 1):
                    switch = True
                    self.clf.eval()

                x_lb, y_lb = Variable(x_lb.cuda()), Variable(y_lb.cuda())
                optimizer.zero_grad()
                #print("StrategyLBTrain " + str(len(x_lb)))



                out_lb, e1 = self.clf(x_lb)

                loss_lb = F.cross_entropy(out_lb, y_lb)
                #accFinal += torch.sum((torch.max(out_lb, 1)[1] == y_lb).float()).data.item()

                if(switch):
                    self.clf.train()
                    switch = False

            else:
                loss_lb = 0

            if (len(x_pl) > 1):

                if(len(x_pl) == 1):
                    switch = True
                    self.clf.train()

                x_pl, y_pl = Variable(x_pl.cuda()), Variable(y_pl.cuda())
                optimizer.zero_grad()
                #print("StrategyPLTrain " + str(len(x_lb)))


                out_pl, e2 = self.clf(x_pl)
                loss_pl = F.cross_entropy(out_pl, y_pl)
                #accFinal += torch.sum((torch.max(out_pl, 1)[1] == y_pl).float()).data.item()

                if(switch):
                    self.clf.train()
                    switch = False

            else:
                loss_pl = 0

            loss = loss_lb + pl_factor * ratiopl * loss_pl

            if(loss > 0):
                loss.backward()

                # clamp gradients, just in case
                for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

                optimizer.step()
        self.clf.eval()
        clb = 0
        cpl = 0
        for batch_idx, (x, y, id, idxs) in enumerate(loader_tr):

           x_lb = x[id == 1]
           y_lb = y[id == 1]

           clb += len(x_lb)
           if (len(x_lb) > 0):
               x_lb, y_lb = Variable(x_lb.cuda()), Variable(y_lb.cuda())
               #print("Strategy " + str(x_lb.shape))
               out_lb, e1 = self.clf(x_lb)
               accFinal += torch.sum((torch.max(out_lb, 1)[1] == y_lb).float()).data.item()

        #print("clb: " + str(clb) + " cpl: " + str(cpl))
        return accFinal/clb

# all training cycles with pseudo-labels
    def trainPL(self, pl_factor, pl_ratio, noearlystop, epochnum):
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

        X_dl = np.concatenate([self.X[idxs_train_lb], self.X[idxs_train_pl]], axis=0)
        Y_dl = np.concatenate([self.Y.numpy()[idxs_train_lb], self.y_pl[idxs_train_pl]], axis=0)
        id_dl = np.concatenate([id_lb, id_pl], axis=0)

        lbpl = len(idxs_train_lb) + len(idxs_train_pl)

        ratiopl = 1
        if(pl_ratio):
            ratiopl = min(1,len(idxs_train_pl)/len(idxs_train_lb))

        #print("comb: " + str(lbpl))
        #print(id_dl)

        loader_tr = DataLoader(self.handler(X_dl, torch.Tensor(Y_dl).long(), id_dl, transform=self.args['transform']),
                               shuffle=True, **self.args['loader_tr_args'])

        epoch = 0
        accCurrent = 0.
        while accCurrent < 0.99 and (epoch < epochnum or noearlystop):
            accCurrent = self._trainPL(epoch, loader_tr, optimizer, pl_factor, ratiopl)
            epoch += 1
            #if (epoch % 50 == 0) and (accCurrent < 0.2):  # reset if not converging
            #    self.clf = self.net.apply(weight_reset)
            #    optimizer = optim.Adam(self.clf.parameters(), lr=self.args['lr'], weight_decay=0)
        print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)
        return accCurrent

# predict classes
    def predict(self, X, Y):
        id = np.empty(len(X))
        id[:] = 1
        if type(X) is np.ndarray:
            loader_te = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']),
                                   shuffle=False, **self.args['loader_te_args'])
        else:
            loader_te = DataLoader(self.handler(X.numpy(), Y, id, transform=self.args['transformTest']),
                                   shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        P = torch.zeros(len(Y)).long()
        with torch.no_grad():
            for x, y, id, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        return P

# predict class probabilities
    def predict_prob(self, X, Y):
        id = np.empty(len(X))
        id[:] = 1
        loader_te = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False,
                               **self.args['loader_te_args'])
        self.clf.eval()
        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        with torch.no_grad():
            for x, y, id, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data

        return probs

# predict class with dropout
    def predict_prob_dropout(self, X, Y, n_drop):
        id = np.empty(len(X))
        id[:] = 1
        loader_te = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                #print('n_drop {}/{}'.format(i + 1, n_drop))
                for x, y, id, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu().data
        probs /= n_drop

        return probs

# predict class probabilities with dropout
    def predict_prob_dropout_split(self, X, Y, n_drop):
        id = np.empty(len(X))
        id[:] = 1
        loader_te = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                #print('n_drop {}/{}'.format(i + 1, n_drop))
                for x, y, id, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
            return probs

    def get_embedding(self, X, Y):
        id = np.empty(len(X))
        id[:] = 1
        loader_te = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, id, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                embedding[idxs] = e1.data.cpu()

        return embedding

    # gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y):
        model = self.clf
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])

        id = np.empty(len(X))
        id[:] = 1
        loader_te = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            for x, y, id, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                cout, out = self.clf(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (
                                        -1 * batchProbs[j][c])
            return torch.Tensor(embedding)


    # measure inconsistency
    def measure_inconsistency(self, loader):

        self.clf.eval()
        uncertainty = np.zeros(len(loader.dataset.X))
        predictions = np.zeros((len(loader.dataset.X), 10, 5))
        # print(uncertainty.shape)
        with torch.no_grad():
            for x, y, id, idxs in loader:
                # print(len(x.numpy()))

                xas1 = Variable(augment(x).cuda())
                xas2 = Variable(augment(x).cuda())
                xas3 = Variable(augment(x).cuda())
                xas4 = Variable(augment(x).cuda())
                x = Variable(x.cuda())

                out0, e0 = self.clf(x)
                prob0 = F.softmax(out0, dim=1).cpu().data[np.newaxis, :, :]

                out1, e1 = self.clf(xas1)
                prob1 = F.softmax(out1, dim=1).cpu().data[np.newaxis, :, :]

                out2, e2 = self.clf(xas2)
                prob2 = F.softmax(out2, dim=1).cpu().data[np.newaxis, :, :]

                out3, e3 = self.clf(xas3)
                prob3 = F.softmax(out3, dim=1).cpu().data[np.newaxis, :, :]

                out4, e4 = self.clf(xas4)
                prob4 = F.softmax(out4, dim=1).cpu().data[np.newaxis, :, :]

                # all probabilities for augmented and base
                merge = np.concatenate([prob0, prob1, prob2, prob3, prob4], axis=0)

                # variance for each class over augmentations (inconsistency of predictions Epsilon)
                variance = np.var(merge, axis=0)

                # sum of variances (aggregate metric C)
                varsum = np.sum(variance, axis=1)

                uncertainty[idxs] = varsum
                predictions[idxs] = np.moveaxis(merge, 0, 2)

        return uncertainty, predictions

# measure various values
    def measure_values(self, loader):

        self.clf.eval()
        uncertainty = np.zeros(len(loader.dataset.X))
        predictions = np.zeros((len(loader.dataset.X), 10, 1))
        with torch.no_grad():
            for x, y, id, idxs in loader:
                # print(len(x.numpy()))

                x = Variable(x.cuda())
                #print("Measure " + str(x.shape))
                out0, e0 = self.clf(x)
                prob0 = F.softmax(out0, dim=1).cpu().data[np.newaxis, :, :]

                # all probabilities for augmented and base
                merge = np.concatenate([prob0], axis=0)
                # print(merge[:,0])

                # variance for each class over augmentations (inconsistency of predictions Epsilon)
                variance = np.var(merge, axis=0)

                # sum of variances (aggregate metric C)
                varsum = np.sum(variance, axis=1)

                uncertainty[idxs] = varsum
                # variances[idxs] = variance
                predictions[idxs] = np.moveaxis(merge, 0, 2)

        return uncertainty, predictions

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
        inconsistency, predictions = self.measure_values(loader)

        predictionsNaN = predictions[:, :, 0]
        predictions = np.nan_to_num(predictionsNaN)
        entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)

        return dict(zip(idxs_pl, inconsistency)), dict(zip(idxs_pl, entropy))

# compute metric over new pseudo-labels
    def testNewPL(self, idxs_pl_new):
        idxs_test = np.arange(self.n_pool)[idxs_pl_new]
        # print(idxs_pl)
        X = self.X[idxs_test]
        # show_img(X[0])
        # show_img(X[1])
        Y = torch.Tensor(self.Y.numpy()[idxs_test]).long()

        id = np.empty(len(X))
        id[:] = 1

        loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False,
                            **self.args['loader_te_args'])
        inconsistency, predictions = self.measure_values(loader)
        predictionsNaN = predictions[:, :, 0]
        predictions = np.nan_to_num(predictionsNaN)
        entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)

        return dict(zip(idxs_test, inconsistency)), dict(zip(idxs_test, entropy))

    # choose true pseudo-labels from candidates
    def choosePL(self, candidates):
        X = self.X[candidates]
        Y = torch.Tensor(self.Y.numpy()[candidates]).long()

        id = np.empty(len(X))
        id[:] = 1

        loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False,
                            **self.args['loader_te_args'])
        self.clf.eval()
        truepl = []
        with torch.no_grad():
            for x, y, id, idxs in loader:
                idxs = idxs.numpy()
                # print(idxs)
                x_sv, y_sv = x,y
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                # print(out)
                probs = F.softmax(out, dim=1)
                probs = probs.cpu().data.numpy()


                if(self.filtertype == 'margin'):
                    probmaxs = np.amax(probs, axis=1)
                    probsecmaxs = np.sort(probs, axis=1)[:, -2]

                    for i in range(len(idxs)):

                        # compare largest probability to second-largest
                        if probmaxs[i] > self.filter_factor * probsecmaxs[i]:
                            truepl.append(idxs[i])
                elif(self.filtertype == 'margintrue'):
                    probmaxs = np.amax(probs, axis=1)
                    probsecmaxs = np.sort(probs, axis=1)[:, -2]

                    for i in range(len(idxs)):

                        # compare largest probability to second-largest
                        if probmaxs[i] - probsecmaxs[i] > self.filter_factor :
                            truepl.append(idxs[i])
                elif(self.filtertype == 'entropy'):
                    entropy = -1 * np.sum(np.multiply(probs, np.nan_to_num(np.log(probs))), axis=1)
                    for i in range(len(idxs)):
                        if entropy[i] < self.filter_factor:
                            truepl.append(idxs[i])

        # print(truepl)
        return truepl

    def updatepl(self, idxs_lb, idxs_pl, y_pl):
        self.idxs_lb = idxs_lb
        self.idxs_pl = idxs_pl
        self.y_pl = y_pl

        idxs = np.arange(self.n_pool)[self.idxs_lb]
        self.store['type'][idxs] = 'lb'
        idxs2 = np.arange(self.n_pool)[self.idxs_pl]
        self.store['type'][idxs2] = 'pl'

        self.store['pseudolabel'][idxs2] = y_pl[idxs2].astype(int)

# class distribution handling
    def getDistributionTrueLB(self):
        idxs_test = np.arange(self.n_pool)[self.idxs_lb]
        Y = torch.Tensor(self.Y.numpy()[idxs_test]).long().numpy()
        x = collections.Counter(Y)

        print(len(idxs_test))
        return sorted(x.items())

    def getDistributionPredictLB(self):
        idxs_test = np.arange(self.n_pool)[self.idxs_lb]

        X = self.X[idxs_test]
        Y = torch.Tensor(self.Y.numpy()[idxs_test]).long().numpy()

        predictions = self.predict(X,Y)

        x = collections.Counter(predictions.numpy())

        print(len(idxs_test))
        return sorted(x.items())

    def getDistributionTruePL(self):
        idxs_test = np.arange(self.n_pool)[self.idxs_pl]


        Y = torch.Tensor(self.Y.numpy()[idxs_test]).long().numpy()
        x = collections.Counter(Y)
        print(len(idxs_test))
        return sorted(x.items())

    def getDistributionPredictPL(self):
        idxs_test = np.arange(self.n_pool)[self.idxs_pl]

        X = self.X[idxs_test]
        Y = torch.Tensor(self.Y.numpy()[idxs_test]).long().numpy()

        predictions = self.predict(X,Y)

        x = collections.Counter(predictions.numpy())

        print(len(idxs_test))
        return sorted(x.items())

    def getDistributionPLPL(self):
        idxs_test = np.arange(self.n_pool)[self.idxs_pl]

        Y = self.y_pl[idxs_test]

        x = collections.Counter(Y)

        print(len(idxs_test))
        return sorted(x.items())

    def getDistributionPLPLPredicted(self):
        idxs_test = np.arange(self.n_pool)[self.idxs_pl]
        X = self.X[idxs_test]
        Y = torch.Tensor(self.Y.numpy()[idxs_test]).long().numpy()

        predictions = self.predict(X,Y)

        Y_pl = self.y_pl[idxs_test]

        plpred = Y_pl[predictions.numpy() == Y_pl]

        x = collections.Counter(plpred)

        print(len(plpred))
        return sorted(x.items())

    def getPLPred(self):
        idxs_test = np.arange(self.n_pool)[self.idxs_pl]
        X = self.X[idxs_test]
        Y = torch.Tensor(self.Y.numpy()[idxs_test]).long().numpy()

        predictions = self.predict(X,Y)

        Y_pl = self.y_pl[idxs_test]

        plpred = Y_pl[predictions.numpy() == Y_pl]

        if(len(X) > 0):
            return len(plpred)/len(X)
        else:
            return 1

    def getDistributionPredicted(self, idxs):
        idxs_test = np.arange(self.n_pool)[idxs]

        X = self.X[idxs_test]
        Y = torch.Tensor(self.Y.numpy()[idxs_test]).long().numpy()

        predictions = self.predict(X,Y)

        x = collections.Counter(predictions.numpy())

        print(len(idxs_test))
        return sorted(x.items())

    def getDistributionTrue(self, idxs):
        idxs_test = np.arange(self.n_pool)[idxs]
        Y = torch.Tensor(self.Y.numpy()[idxs_test]).long().numpy()
        x = collections.Counter(Y)

        print(len(idxs_test))
        return sorted(x.items())

    def getDistributionPL(self, idxs):
        idxs_test = np.arange(self.n_pool)[idxs]

        Y = self.y_pl[idxs_test]

        x = collections.Counter(Y)

        print(len(idxs_test))
        return sorted(x.items())

# logging of pseudo-label avlues to file
    def logToFile(self, path):
        logfile = open(path, "w")
        logfile.write(self.store.to_string())
        logfile.close()


    def logPredictions(self):
        id = np.empty(len(self.X))
        id[:] = 1
        if type(self.X) is np.ndarray:
            loader_te = DataLoader(self.handler(self.X, self.Y, id, transform=self.args['transformTest']),
                                   shuffle=False, **self.args['loader_te_args'])
        else:
            loader_te = DataLoader(self.handler(self.X.numpy(), self.Y, id, transform=self.args['transformTest']),
                                   shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        P = torch.zeros(len(self.Y)).long()
        with torch.no_grad():
            for x, y, id, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        self.store['prediction'] = P

    def updatefilter(self,newfilter):
        self.filter_factor = newfilter


