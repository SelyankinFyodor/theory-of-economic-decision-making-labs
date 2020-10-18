import numpy as np
import math


def s_m(n, p, sample, mean):
	s = np.zeros((p, p))

	for i in range(p):
		for j in range(p):
			s[i][j] = sum([(el[0] - mean[i]) * (el[1] - mean[j]) for el in zip(sample[i], sample[j])]) / (n - 1)

	return s


class BayesClassifier:
	def __init__(self):
		self.a = []
		self.q1 = 0
		self.q2 = 0
		self.c = 0
		self.d = 0
		self.dh = 0
		self.p12 = 0
		self.p21 = 0

	def train(self, s1, s2):
		n1 = s1.shape[0]
		p1 = s1.shape[1]
		n2 = s2.shape[0]
		p2 = s2.shape[1]
		mean1 = np.mean(s1, axis=0)
		mean2 = np.mean(s2, axis=0)

		s = ((n1 - 1) * s_m(n1, p1, s1.transpose(), mean1) + (n2 - 1) * s_m(n2, p2, s2.transpose(), mean2)) / (
				n1 + n2 - 2)

		self.a = np.linalg.inv(s).dot(mean1 - mean2)

		z1 = np.mean([self.a.dot(s1[i, :]) for i in range(n1)])
		z2 = np.mean([self.a.dot(s2[i, :]) for i in range(n2)])

		self.q1 = n1 / (n1 + n2)
		self.q2 = n2 / (n1 + n2)

		self.c = (z1 + z2) / 2 + math.log(self.q1 / self.q2)

		summa = (self.a.dot(s)).dot(self.a)
		self.d = (z1 - z2) * (z1 - z2) / summa
		self.dh = (n1 + n2 - p1 - 3) * self.d / (n1 + n2 - 2) - p1 * (1 / n1 + 1 / n2)
		k = math.log(self.q2 / self.q1)
		self.p21 = 0.5 * (1 + math.erf(((- k - self.dh / 2) / math.sqrt(self.d)) / math.sqrt(2)))
		self.p12 = 0.5 * (1 + math.erf(((k - self.dh / 2) / math.sqrt(self.d)) / math.sqrt(2)))

	def get_class(self, vector, is_q_set=True):
		if is_q_set:
			return 1 if vector.dot(self.a) < self.c + math.log(self.q1 / self.q2) else 0
		else:
			return 1 if vector.dot(self.a) < self.c else 0

	def classify(self, sample1, sample2):
		q1 = 0
		q2 = 0

		for i in range(sample1.shape[0]):
			x = sample1[i]
			if self.get_class(x) == 1:
				q1 += 1

		for i in range(sample2.shape[0]):
			x = sample2[i]
			if self.get_class(x) == 0:
				q2 += 1
		return q1, q2

	def print(self):
		print('c')
		print(self.c)
		print('p12')
		print(self.p12)
		print('p21')
		print(self.p21)
		print('q1')
		print(self.q1)
		print('q2')
		print(self.q2)
		print('d')
		print(self.d)
		print('dh')
		print(self.dh)
