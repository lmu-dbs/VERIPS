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

# BALD without pseudo-labels
class BALDDropout(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDDropout, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype)
		self.n_drop = n_drop

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob_dropout_split(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], self.n_drop)
		pb = probs.mean(0)
		entropy1 = (-pb*torch.log(pb)).sum(1)
		entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
		U = entropy2 - entropy1
		#print(U.sort()[0][:n])
		return idxs_unlabeled[U.sort()[1][:n]]


# BALD with Consistency-based pseudo-labels
class BALDDropoutConsistencyPL(BALDDropout):

	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDDropoutConsistencyPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop)
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
			idxs = np.arange(self.n_pool)[idxs_unlabeled]
			self.store['type'][idxs] = 'plcand'
			chosen = self.choosePL(idxs_unlabeled)
			idxs2 = np.arange(self.n_pool)[idxs_unlabeled[chosen]]
			self.store['type'][idxs2] = 'plchosen'

			return idxs_unlabeled[chosen]

# BALD architecture with random active learning
class BALDDropoutRand(BALDDropoutConsistencyPL):
	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDDropoutRand, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop)
		self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
		self.y_pl = np.zeros(self.n_pool)

	def query(self, n):
		inds = np.where(self.idxs_lb == 0)[0]
		return inds[np.random.permutation(len(inds))][:n]

# BALD with random pseudo-labels
class BALDDropoutRandPL(BALDDropoutConsistencyPL):
	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDDropoutRandPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop)
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


# BALD with Entropy-based pseudo-labels
class BALDDropoutEntropyPL(BALDDropoutConsistencyPL):
	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDDropoutEntropyPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop)
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

# BALD with Margin-based pseudo-labels
class BALDDropoutMarginPL(BALDDropoutConsistencyPL):
	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDDropoutMarginPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop)
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

# BALD with Gram-based pseudo-labels
class BALDDropoutGradientPL(BALDDropoutConsistencyPL):
	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDDropoutGradientPL, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop)
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

# BALD with Gradient-based pseudo-labels
class BALDDropoutPL3(BALDDropoutConsistencyPL):
	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDDropoutPL3, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype,
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

# BALD with BALD-based pseudo-labels
class BALDDropoutBALDInv(BALDDropoutConsistencyPL):
	def __init__(self, X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop=10):
		super(BALDDropoutBALDInv, self).__init__(X, Y, idxs_lb, net, handler, args, filter_factor, filtertype, n_drop)
		self.idxs_pl = np.zeros(self.n_pool, dtype=bool)
		self.y_pl = np.zeros(self.n_pool)

	def queryPL(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~np.logical_or(self.idxs_lb, self.idxs_pl)]
		if len(idxs_unlabeled) >= n:
			candidates = self.choosePLCandidates(n, idxs_unlabeled)
			chosen = self.choosePL(idxs_unlabeled[candidates])
			return idxs_unlabeled[candidates[chosen]]
		else:
			idxs = np.arange(self.n_pool)[idxs_unlabeled]
			self.store['type'][idxs] = 'plcand'
			if(self.filtertype == 'bald'):
				chosen = self.choosePL2(idxs_unlabeled)
			else:
				chosen = self.choosePL(idxs_unlabeled)
			idxs2 = np.arange(self.n_pool)[idxs_unlabeled[chosen]]
			self.store['type'][idxs2] = 'plchosen'

			return idxs_unlabeled[chosen]

	def choosePLCandidates(self, n, idxs_unlabeled):
		X = self.X[idxs_unlabeled]
		Y = torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long()

		probs = self.predict_prob_dropout_split(X, Y, self.n_drop)
		pb = probs.mean(0)
		entropy1 = np.nan_to_num((-pb*torch.log(pb)).sum(1))
		entropy2 = np.nan_to_num((-probs*torch.log(probs)).sum(2).mean(0))
		U = entropy2 - entropy1

		almax = np.argpartition(U, -n)[-n:]
		#print(U[almax])

		return almax

	def choosePL2(self, idxs_unlabeled):
		X = self.X[idxs_unlabeled]
		Y = torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long()

		probs = self.predict_prob_dropout_split(X, Y, self.n_drop)
		pb = probs.mean(0)
		entropy1 = np.nan_to_num((-pb*torch.log(pb)).sum(1))
		entropy2 = np.nan_to_num((-probs*torch.log(probs)).sum(2).mean(0))
		U = entropy2 - entropy1

		inds = []
		for i in range(len(idxs_unlabeled)):
			if U[i] > self.filter_factor:
				inds.append(i)

		return inds