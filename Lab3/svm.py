import numpy as np
import cvxopt
from qpsolvers import solve_qp


def svm(train, classes, kern, C, threshold):
	n_samples, n_features = train.shape
	K = np.zeros((n_samples, n_samples))
	for i in range(n_samples):
		for j in range(n_samples):
			K[i, j] = kern(train[i], train[j])
	P = cvxopt.matrix(np.outer(classes, classes) * K)
	q = cvxopt.matrix(np.ones(n_samples) * -1)
	A = cvxopt.matrix(classes, (1, n_samples))
	b = cvxopt.matrix(.0)
	G = cvxopt.matrix(
		np.concatenate((np.diag(-np.ones(n_samples)), np.diag(np.ones(n_samples))), axis=0)
	)
	h = cvxopt.matrix(
		np.concatenate((np.zeros(n_samples), np.full(n_samples, C)), axis=0)
	)
	solution = cvxopt.solvers.qp(P, q, G, h, A, b)
	a = np.ravel(solution['x'])
	sv = a > threshold
	ind = np.arange(len(a))[sv]
	a = a[sv]
	sv_train = train[sv]
	sv_classes = classes[sv]
	b = 0
	for n in range(len(a)):
		b += sv_classes[n]
		b -= np.sum(a * sv_classes * K[ind[n], sv])
	b /= len(a)
	w = np.zeros(n_features)
	for n in range(len(a)):
		w += a[n] * sv_classes[n] * sv_train[n]
	return a, ind, w, b, sv_train, sv_classes
