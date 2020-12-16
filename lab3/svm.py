import numpy as np
import cvxopt


def svm(train, classes, kern, C, threshold, part=3):
	n_samples, n_features = train.shape
	K = np.zeros((n_samples, n_samples))
	for i in range(n_samples):
		for j in range(n_samples):
			K[i, j] = kern(train[i], train[j])
	P = cvxopt.matrix(np.outer(classes, classes) * K)
	q = cvxopt.matrix(-np.ones(n_samples))
	A = cvxopt.matrix(classes, (1, n_samples))
	b = cvxopt.matrix(.0)
	G = cvxopt.matrix(
		np.concatenate((np.diag(-np.ones(n_samples)), np.diag(np.ones(n_samples))), axis=0)
	)
	h = cvxopt.matrix(
		np.concatenate((np.zeros(n_samples), np.full(n_samples, C)), axis=0)
	)
	solver = cvxopt.solvers
	solver.options['show_progress'] = False
	a = np.ravel(solver.qp(P, q, G, h, A, b)['x'])
	sv = a > threshold
	ind = np.arange(len(a))[sv]
	if part == 1 or part == 2 or part == 4:
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
	if part == 3:
		sv_x = []
		sv_classes = []
		for index in ind:
			sv_x.append(train[index])
			sv_classes.append(classes[index])
		sv_x = np.array(sv_x)
		return a, sv_x, sv_classes


def decision_func(x1, x2, n_samples, a, y, kern):
	res = 0
	for i in range(n_samples):
		res += a[i] * y[i] * kern(x1[i], x2)
	return res


def predict_simple(x, w, b):
	return 1 if w.dot(x) + b > 0 else -1


def predict(sample, w, b):
	classes = []
	for x in sample:
		classes.append((1 if w.dot(x) + b > 0 else -1))
	return np.array(classes)
