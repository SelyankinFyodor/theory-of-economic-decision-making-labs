import numpy as np
from matplotlib import pyplot as plt
import math
from Lab3.svm import svm, decision_func


def lin_kern(x, y):
	return x.dot(y)


def poly_kern(x, y):
	return (1+x.dot(y)).dot((1+x.dot(y))).dot((1+x.dot(y)))


def gauss_kernel(x1, x2, sigma=1):
	return math.exp(-np.linalg.norm(x1-x2) / (2 * sigma ** 2))


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
		plt.plot(x, y_border_1, c+'-.')
		plt.plot(x, y_border_2, c+'-.')
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
		plt.scatter(data['sv_x'][:, 0], data['sv_x'][:, 1], c=data['sv_y'], marker='x', cmap='coolwarm', label="support vectors")
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
		alpha, sv_x, sv_classes = svm(train=A, classes=classes, kern=gauss_kernel, C=C,
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
			z_grid[i] = decision_func(x1=A, x2=xy[i], a=alpha, n_samples=size, y=classes, kern=gauss_kernel)

		plt.figure(1)
		Z = z_grid.reshape(xx.shape)
		plt.contourf(xx, yy, Z, alpha=.8)
		plt.scatter(A[:, 0], A[:, 1], c=classes, marker='.', cmap='coolwarm')
		plt.scatter(sv_x[:, 0], sv_x[:, 1], c=sv_classes, marker='x', cmap='coolwarm')
		plt.title(f"C={C}")
		plt.grid(True)
		plt.show()


def main4():
	return


def main5():
	return


if __name__ == "__main__":
	# main1()
	# main2()
	# main3()
	main4()
