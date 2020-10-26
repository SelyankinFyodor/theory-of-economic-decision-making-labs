import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
from sklearn import datasets


def main2d():
	means = (0, 0)
	cov = np.array([[1.9, 1.2], [1.2, 1.9]])
	A = np.random.multivariate_normal(means, cov, 100).transpose()
	C = normalize(A)
	eig_value, eig_vectors = np.linalg.eig(np.cov(C))
	Z = eig_vectors.T.dot(C)
	print(np.cov(Z))
	fig = plt.figure(1)
	ax = fig.add_subplot()
	ax.scatter(C[0], C[1], c='b', marker='.')
	len0 = math.sqrt(math.fabs(eig_value[0]))
	len1 = math.sqrt(math.fabs(eig_value[1]))
	t = np.linspace(0, 1, 100)
	x_v = t * eig_vectors[0][0] * len0
	y_v = t * eig_vectors[1][0] * len0
	ax.plot(x_v, y_v, linewidth=5, c='r')
	x_v = t * eig_vectors[0][1] * len1
	y_v = t * eig_vectors[1][1] * len1
	ax.plot(x_v, y_v, linewidth=5, c='r')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_title('данные и собсвенные вектора матрицы ковариации')
	plt.show()


def main3d():
	means = (0, 0, 0)
	cov = np.array([[7, 5, 3], [5, 9, 3], [3, 3, 5]])
	A = np.random.multivariate_normal(means, cov, 100).transpose()
	C = normalize(A)
	print(np.cov(C))
	print(np.mean(C, axis=1))
	eig_value, eig_vectors = np.linalg.eig(np.cov(C))
	Z = eig_vectors.T.dot(C)
	print(np.cov(Z))
	fig = plt.figure(1)
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(C[0], C[1], C[2], c='b', marker='.')
	len0 = math.sqrt(math.fabs(eig_value[0]))
	len1 = math.sqrt(math.fabs(eig_value[1]))
	len2 = math.sqrt(math.fabs(eig_value[2]))
	t = np.linspace(0, 1, 100)
	x_ev = t * eig_vectors[0][0] * len0
	y_ev = t * eig_vectors[1][0] * len0
	z_ev = t * eig_vectors[2][0] * len0
	ax.plot(x_ev, y_ev, z_ev, linewidth=1, c='r')
	x_ev = t * eig_vectors[0][1] * len1
	y_ev = t * eig_vectors[1][1] * len1
	z_ev = t * eig_vectors[2][1] * len1
	ax.plot(x_ev, y_ev, z_ev, linewidth=1, c='r')
	x_ev = t * eig_vectors[0][2] * len2
	y_ev = t * eig_vectors[1][2] * len2
	z_ev = t * eig_vectors[2][2] * len2
	ax.plot(x_ev, y_ev, z_ev, linewidth=1, c='r')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.set_title('данные и собсвенные вектора матрицы ковариации')
	plt.show()


def german_process():
	n = 1000
	p = 24
	matrix = np.loadtxt('../data/german.data-numeric')
	cols = [i for i in range(p)]
	classes = []
	col_i = 0
	A = np.zeros((p, n))
	for i in range(n):
		for j in range(p):
			A[j][col_i] = matrix[i][cols[j]]
		col_i += 1
		classes.append(matrix[i][p])
	A = StandardScaler().fit_transform(A.transpose()).transpose()
	eig_value, eig_vectors = np.linalg.eig(np.cov(A))
	Z = eig_vectors.T.dot(A)
	print(np.cov(Z))
	fig = plt.figure(2)
	ax = fig.add_subplot(111, projection='3d')
	colors = ['r', 'b']
	cov_x = np.cov(A)
	cov_z = np.cov(Z)
	d_x = []
	d_z = []
	for i in range(p):
		d_x.append(cov_x[i][i])
		d_z.append(cov_z[i][i])
	for i in range(n):
		ax.scatter(Z[0][i], Z[1][i], Z[2][i], c=colors[int(classes[i] - 1)], marker='.')
	[print("{:.5f}".format(f), end=' ') for f in sorted(d_x, key=lambda x: -x)]
	print()
	[print("{:.5f}".format(f), end=' ') for f in sorted(d_z, key=lambda x: -x)]
	print()
	print(sum(d_z))
	plt.title('данные german')
	plt.show()


def iris_process():
	iris = datasets.load_iris()
	A = StandardScaler().fit_transform(X=np.array(iris.data[:, :])).transpose()
	classes = iris.target
	eig_value, eig_vectors = np.linalg.eig(np.cov(A))
	Z = eig_vectors.T.dot(A)
	points = Z.transpose()

	print("A:")
	print(np.cov(A))
	print(np.cov(Z))
	mtr = np.identity(4)
	for i in range(4):
		mtr[i][i] = math.sqrt(eig_value[i])
	print(eig_vectors.dot(mtr))

	colors = ['r', 'g', 'b']
	markers = ['*', 'o', '^']
	for i in range(points.shape[0]):
		x = points[i]
		c = classes[i]
		plt.plot(x[0], x[1], f'{colors[c]}{markers[c]}')
	plt.title('данные Iris')
	plt.show()


if __name__ == "__main__":
	main2d()
	main3d()
	german_process()
	iris_process()
