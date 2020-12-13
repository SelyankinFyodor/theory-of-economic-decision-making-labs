import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from Lab3.svm import svm


def lin_kern(x, y):
	return x.dot(y)


def rbf_kern(x, y):
	return np.exp(-cdist(x, y)/2)


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
	alpha, sv_ids, w, b, sv_x, sv_y = svm(train=A12.transpose(), classes=classes, kern=lin_kern, C=100000,
		threshold=1e-3)

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
	plt.grid(True)
	plt.show()

	print("w:", w)
	print(2 / np.linalg.norm(w))


if __name__ == "__main__":
	main1()
