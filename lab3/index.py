from matplotlib import pyplot as plt
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from lab3.svm import svm, decision_func, predict


def lin_kern(x, y):
	return x.dot(y)


def poly_kern(x, y):
	return (1 + x.dot(y)).dot((1 + x.dot(y))).dot((1 + x.dot(y)))


def get_gauss(sigma=1):
	return lambda x1, x2: math.exp(-np.linalg.norm(x1 - x2) / (2 * sigma ** 2))


def cv(X, y, parts=5):
	part_len = X.shape[0] // parts
	yield X[part_len:, :], X[:part_len, :], y[part_len:], y[:part_len]

	for i in range(1, parts - 1):
		train_start = i * part_len
		train_end = (i + 1) * part_len
		yield np.concatenate((X[:train_start, :], X[train_end:, :]), axis=0), \
		      X[train_start:train_end, :], \
		      np.concatenate((y[:train_start], y[train_end:]), axis=0), \
		      y[train_start:train_end]
	yield X[:len(X) - part_len, :], X[len(X) - part_len:, :], y[:len(y) - part_len], y[len(y) - part_len:]


def main1():
	m1 = (2, 2)
	m2 = (-1, -1)
	cov = [
		[1, -0.7],
		[-0.7, 1]
	]
	size1, size2 = 50, 50
	A1 = np.random.multivariate_normal(m1, cov, size1).transpose()
	A2 = np.random.multivariate_normal(m2, cov, size2).transpose()
	A12 = np.concatenate((A1, A2), axis=1)
	classes = np.array([-1.0 if i >= size1 else 1.0 for i in range(size1 + size2)])
	alpha, sv_ids, w, b, sv_x, sv_y = svm(
		train=A12.transpose(),
		classes=classes,
		kern=lin_kern,
		C=100000,
		threshold=1e-3, part=1)

	plt.figure(2)
	x = np.linspace(-4, 5, 10)
	y = (-b - w[0] * x) / w[1]
	y_border_1 = (-b - 1 - w[0] * x) / w[1]
	y_border_2 = (-b + 1 - w[0] * x) / w[1]

	plt.plot(x, y, 'k-')
	plt.plot(x, y_border_1, 'r--')
	plt.plot(x, y_border_2, 'b--')
	plt.scatter(A12.transpose()[:, 0], A12.transpose()[:, 1], c=classes, marker='.', cmap='coolwarm',
		label="the elements of a sample")
	plt.scatter(sv_x[:, 0], sv_x[:, 1], c=sv_y, marker='x', cmap='coolwarm', label="support vectors")
	plt.legend()
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('well separated data')
	plt.grid(True)
	plt.show()

	print("w:", w)
	print(2 / np.linalg.norm(w))


def main2():
	m1 = (2, 2)
	m2 = (-1, -1)
	cov = [
		[1, 0.7],
		[0.7, 1]
	]
	size1, size2 = 50, 50
	A1 = np.random.multivariate_normal(m1, cov, size1).transpose()
	A2 = np.random.multivariate_normal(m2, cov, size2).transpose()
	A12 = np.concatenate((A1, A2), axis=1)
	classes = np.array([-1.0 if i >= size1 else 1.0 for i in range(size1 + size2)])
	join = []
	x = np.linspace(-4, 5, 10)
	for C, c in zip([1, 10, 100, 1000], ['c', 'g', 'm', 'y']):
		alpha, sv_ids, w, b, sv_x, sv_y = svm(
			train=A12.transpose(),
			classes=classes,
			kern=lin_kern,
			C=C,
			threshold=1e-3,
			part=1)
		print("w:", w)
		print(2 / np.linalg.norm(w))
		y = (-b - w[0] * x) / w[1]
		y_border_1 = (-b - 1 - w[0] * x) / w[1]
		y_border_2 = (-b + 1 - w[0] * x) / w[1]
		join.append({'y': y, 'b1': y_border_1, 'b2': y_border_2, 'sv_x': sv_x, 'sv_y': sv_y})
		plt.figure(2)
		plt.scatter(A12.transpose()[:, 0], A12.transpose()[:, 1], c=classes, marker='.', cmap='coolwarm',
			label="the elements of a sample")
		plt.plot(x, y, 'k-')
		plt.plot(x, y_border_1, c + '-.')
		plt.plot(x, y_border_2, c + '-.')
		plt.scatter(sv_x[:, 0], sv_x[:, 1], c=sv_y, marker='x', cmap='coolwarm', label="support vectors")
		plt.xlabel('x')
		plt.ylabel('y')
		plt.title(f'bad separated data, c={C}')
		plt.grid(True)
		plt.show()

	plt.figure(2)
	plt.scatter(A12.transpose()[:, 0], A12.transpose()[:, 1], c=classes, marker='.', cmap='coolwarm',
		label="the elements of a sample")
	for data, c in zip(join, ['c', 'g', 'm', 'y']):
		plt.plot(x, data['y'], 'k-')
		plt.plot(x, data['b1'], c + '-.')
		plt.plot(x, data['b2'], c + '-.')
		plt.scatter(data['sv_x'][:, 0], data['sv_x'][:, 1], c=data['sv_y'], marker='x', cmap='coolwarm',
			label="support vectors")
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title(f'bad separated data')
	plt.grid(True)
	plt.show()


def main3():
	m1 = (0, 0)
	cov = [
		[1, 0],
		[0, 1]
	]
	size = 100
	A = np.random.multivariate_normal(m1, cov, size)
	classes = np.array([-1.0 if (A[i][0] > 0) ^ (A[i][1] > 0) else 1.0 for i in range(size)])
	for C in [1, 10, 100, 1000]:
		alpha, sv_x, sv_classes = svm(train=A, classes=classes, kern=get_gauss(), C=C,
			threshold=1e-3)
		h = .1
		x_min, x_max = A[:, 0].min() - .5, A[:, 0].max() + .5
		y_min, y_max = A[:, 1].min() - .5, A[:, 1].max() + .5
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

		x_grid = xx.ravel()
		y_grid = yy.ravel()
		xy = np.c_[x_grid, y_grid]

		z_grid = np.zeros(x_grid.shape[0])
		for i in range(x_grid.shape[0]):
			z_grid[i] = decision_func(x1=A, x2=xy[i], a=alpha, n_samples=size, y=classes, kern=get_gauss())

		plt.figure(1)
		Z = z_grid.reshape(xx.shape)
		plt.contourf(xx, yy, Z, alpha=.8)
		plt.scatter(A[:, 0], A[:, 1], c=classes, marker='.', cmap='coolwarm')
		plt.scatter(sv_x[:, 0], sv_x[:, 1], c=sv_classes, marker='x', cmap='coolwarm')
		plt.title(f"C={C}")
		plt.grid(True)
		plt.show()


def main4(withPCA=False):
	matrix = np.loadtxt('../data/german.data-numeric')
	X = matrix[:, :24]
	y = 2 * matrix[:, 24] - 3
	X, y = shuffle(X, y)
	c_sample = StandardScaler().fit_transform(X.transpose()).transpose()
	if withPCA:
		eig_value, eig_vectors = np.linalg.eig(np.cov(c_sample.transpose()))
		Z = eig_vectors.T.dot(c_sample.transpose())
		Z = Z[:3, :]
		c_sample = StandardScaler().fit_transform(Z.transpose())
	X_train, X_test, y_train, y_test = train_test_split(c_sample, y, test_size=0.2, random_state=0)
	best = {'sigma': 1, 'C': 1, 'er_p': 1}
	for C in [1, 10, 100, 1000]:
		for sigma in [1, 10, 100]:
			er_av = []
			for X_cv_train, X_cv_test, y_cv_train, y_cv_test in cv(X_train, y_train, parts=4):
				a, ind, w, b, sv_train, sv_classes = \
					svm(X_cv_train, y_cv_train, kern=get_gauss(sigma), C=C, part=4, threshold=1e-3)
				predicted_classes = predict(X_cv_test, w, b)
				errors = np.count_nonzero(predicted_classes != y_cv_test)
				error_probability = errors / y_cv_test.shape[0]
				print(f"c={C} sigma={sigma} error={error_probability}")
				er_av += [error_probability]
			ave = np.average(er_av)
			print(f"c={C} sigma={sigma} error={ave}")
			if ave < best['er_p']:
				best = {
					'sigma': sigma,
					'C': C,
					'er_p': ave
				}
	print(f"best: c={best['C']} sigma={best['sigma']} error={best['er_p']}")
	a, ind, w, b, sv_train, sv_classes = \
		svm(X_train, y_train, kern=get_gauss(best['sigma']), C=best['C'], part=4, threshold=1e-3)
	predicted_classes = predict(X_test, w, b)
	errors = np.count_nonzero(predicted_classes != y_test)
	error_probability = errors / y_test.shape[0]
	print(errors)
	print(error_probability)


def main5():
	return


if __name__ == "__main__":
	# main1()
	# main2()
	# main3()
	# main4()
	# main4(withPCA=True)
	main5()
