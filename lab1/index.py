import numpy as np
import os
from bayes import BayesClassifier as BC


def main1():
	root_path = os.path.abspath("../")
	data_path = os.path.join(root_path, 'data/german.data')
	matrix = np.loadtxt(data_path)
	sample1 = parse_file(matrix, 700, marker=1, p=24)
	sample2 = parse_file(matrix, 300, marker=2, p=24)
	bc = BC()
	start_pos_x = 0
	start_pos_y = 0
	x_size = 200
	y_size = 200
	x_test_pos = 200
	y_test_pos = 200
	x_t_size = 100
	y_t_size = 100
	train_x = sample1[start_pos_x:start_pos_x + x_size]
	train_y = sample2[start_pos_y:start_pos_y + y_size]
	test_x = sample1[x_test_pos:x_test_pos + x_t_size]
	test_y = sample2[y_test_pos:y_test_pos + y_t_size]

	bc.train(train_x, train_y)
	bc.print()
	print(bc.classify(train_x, train_y))
	print(bc.classify(test_x, test_y))


def main2():
	cov = [[5, 1, 2],
	       [1, 7, 4],
	       [2, 4, 3]]

	mean1 = (0, 0, 0)
	mean2 = (8, 8, 8)
	X = np.random.multivariate_normal(mean1, cov, 200)
	Y = np.random.multivariate_normal(mean2, cov, 200)
	bc = BC()
	bc.train(X, Y)
	bc.print()
	print(bc.classify(X, Y))
	X_T = np.random.multivariate_normal(mean1, cov, 100)
	Y_T = np.random.multivariate_normal(mean2, cov, 100)
	print(bc.classify(X_T, Y_T))


def parse_file(matrix, n, marker, p=24):
	sample = np.zeros((n, p))
	cnt = 0
	i = 0
	while cnt < n:
		if matrix[i][p] == marker:
			sample[cnt] = matrix[i][:p]
			cnt += 1
		i += 1
		if i >= n:
			i = 0

	return sample


main2()
