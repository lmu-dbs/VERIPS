import numpy as np
import torch
from .strategy import Strategy
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
import random
from torch import nn
import torch.optim as optim

from scipy import stats
from sklearn.metrics import pairwise_distances
import pdb


# BALD without Dropout in the architecture

class BALDND(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDND, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
		self.n_drop = n_drop

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob_dropout_split(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], self.n_drop)
		pb = probs.mean(0)
		entropy1 = (-pb*torch.log(pb)).sum(1)
		entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
		U = entropy2 - entropy1
		return idxs_unlabeled[U.sort()[1][:n]]

	def _trainPL2(self, epoch, loader_tr, optimizer, pl_factor):
		self.clf.train()
		accFinal = 0.
		clb = 0
		cpl = 0
		for batch_idx, (x, y, id, idxs) in enumerate(loader_tr):

			x_lb = x[id == 1]
			y_lb = y[id == 1]
			x_pl = x[id == 2]
			y_pl = y[id == 2]

			clb += len(x_lb)
			cpl += len(x_pl)
			if (len(x_lb) > 0):
				x_lb, y_lb = Variable(x_lb.cuda()), Variable(y_lb.cuda())
				optimizer.zero_grad()
				#print("StrategyLBTrain " + str(len(x_lb)))
				out_lb, e1 = self.clf(x_lb)
				loss_lb = F.cross_entropy(out_lb, y_lb)
				#accFinal += torch.sum((torch.max(out_lb, 1)[1] == y_lb).float()).data.item()
			else:
				loss_lb = 0

			if (len(x_pl) > 0):
				x_pl, y_pl = Variable(x_pl.cuda()), Variable(y_pl.cuda())
				optimizer.zero_grad()
				#print("StrategyPLTrain " + str(len(x_lb)))
				out_pl, e2 = self.clf(x_pl)
				loss_pl = F.cross_entropy(out_pl, y_pl)
				#accFinal += torch.sum((torch.max(out_pl, 1)[1] == y_pl).float()).data.item()
			else:
				loss_pl = 0

			loss = loss_lb + pl_factor * loss_pl

			loss.backward()

			# clamp gradients, just in case
			for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

			optimizer.step()


	def trainPL2(self, pl_factor):
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
		# print("comb: " + str(lbpl))
		# print(id_dl)

		loader_tr = DataLoader(self.handler(X_dl, torch.Tensor(Y_dl).long(), id_dl, transform=self.args['transform']),
							   shuffle=True, **self.args['loader_tr_args'])

		epoch = 0
		accCurrent = 0.
		while accCurrent < 0.99 and epoch < 50:
			self._trainPL(epoch, loader_tr, optimizer, pl_factor)

			P = self.predict(self.X[idxs_train_lb], self.Y.numpy()[idxs_train_lb]).numpy()

			#print(len(P))
			#print(type(P))
			#print(self.Y.numpy()[idxs_train_lb] == P)

			accCurrent = 1.0 * (self.Y.numpy()[idxs_train_lb] == P).sum().item() / len(idxs_train_lb)

			epoch += 1
			#if (epoch % 50 == 0) and (accCurrent < 0.2):  # reset if not converging
			#	self.clf = self.net.apply(weight_reset)
			#	optimizer = optim.Adam(self.clf.parameters(), lr=self.args['lr'], weight_decay=0)
		print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)
		return accCurrent

class BALDNDConsistencyPL(BALDND):

	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDNDConsistencyPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop)
		self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
		self.y_pl = np.zeros(self.n_pool)

	def choosePLCandidates(self, n, idxs_unlabeled):

		X = self.X[idxs_unlabeled]
		Y = torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long()

		id = np.empty(len(X))
		id[:] = 1

		loader = DataLoader(self.handler(X, Y, id, transform=self.args['transformTest']), shuffle=False,
							**self.args['loader_te_args'])
		inconsistency, predictions = self.measure_inconsistency(loader)

		predictions = predictions[:, :, 0]

		almin = np.argpartition(inconsistency, n)[:n]

		entropy = -1 * np.sum(np.multiply(predictions, np.nan_to_num(np.log(predictions))), axis=1)
		print("EntropyPLCand " + str(np.mean(entropy[almin])))

		return almin

	def queryPL(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~np.logical_or(self.idxs_lb, self.idxs_pl)]
		if len(idxs_unlabeled) >= n:
			candidates = self.choosePLCandidates(n, idxs_unlabeled)
			chosen = self.choosePL(idxs_unlabeled[candidates])
			return idxs_unlabeled[candidates[chosen]]
		else:
			return idxs_unlabeled[[]]


class BALDNDRand(BALDNDConsistencyPL):
	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDNDRand, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop)
		self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
		self.y_pl = np.zeros(self.n_pool)

	def query(self, n):
		inds = np.where(self.idxs_lb == 0)[0]
		return inds[np.random.permutation(len(inds))][:n]

class BALDNDRandPL(BALDNDConsistencyPL):
	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDNDRandPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop)
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



class BALDNDEntropyPL(BALDNDConsistencyPL):
	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDNDEntropyPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop)
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


class BALDNDMarginPL(BALDNDConsistencyPL):
	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDNDMarginPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop)
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

	L_aug = np.block([[L_Y, b_u], [b_u.T, c_u]])
	L_aug_inv = np.block([[L_Y_inv + g_u.dot(g_u.T / d_u), -g_u / d_u], [-g_u.T / d_u, 1.0 / d_u]])

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

		p = min(1, (c_v - b_v.T.dot(L_Y_inv.dot(b_v))) / (c_u - b_u.T.dot(L_Y_inv.dot(b_u))))

		if rng.uniform() <= 1 - p:
			X[u] = False
			X[v] = True
			Ind = Ind_red + [v]
			L_X, L_X_inv = gram_aug(L_Y, L_Y_inv, b_v, c_v)

	# if i % k == 0:
	# print('Iter ', i)

	return Ind


class BALDNDGradientPL(BALDNDConsistencyPL):
	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDNDGradientPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop)
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

# kmeans ++ initialization
def init_centers(X, K):
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
	return indsAll

class BALDNDPL3(BALDNDConsistencyPL):
	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDNDPL3, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype,
											 n_drop)
		self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
		self.y_pl = np.zeros(self.n_pool)

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

		return inds