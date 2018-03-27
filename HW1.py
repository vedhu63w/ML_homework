from heapq				import heappush, heappushpop
from matplotlib			import pyplot as plt
from numpy 				import concatenate, eye, zeros, sqrt
from numpy.random		import multivariate_normal, rand



def generate_data(dim, data_size):
	mean_1 = [0] 
	cov_1 = [[1]]
	mean_2 = [3]; mean_2.extend(zeros(dim-1))
	cov_2 = eye(dim, dtype=int)
	x, y = [], []
	for i in range(data_size):
		if (rand() > 0.5):
			new_x = multivariate_normal(mean_1, cov_1)
			new_x = concatenate((new_x, zeros(dim-1)))
			y_new = 0
		else:
			new_x = multivariate_normal(mean_2, cov_2)
			y_new = 1
		x.append(new_x)
		y.append(y_new)
	return x, y


def L2_dist(a, b):
	assert len(a) == len(b)
	val = 0
	for dim in range(len(a)):
		val += (a[dim] - b[dim])**2
	return sqrt(val)


def nearest_neigh(k, test_point, data):
	# heap pop returns minimum value. we want to be able to return max value. so adding "-"
	heap = []
	for i in range(k): heappush(heap, (-float("inf"), -1))
	for src_point, src_label in data:
		curr_dist = L2_dist(test_point, src_point)
		heappushpop(heap, (-curr_dist, src_label))				# luckily python compares tuples and does element wise starting from first element
	return heap


def do_kNN(k, test_x, test_y, train_x, train_y):
	error_01 = 0
	train_xy = zip(train_x, train_y)
	for test_point, label in zip(test_x, test_y):
		lab_sum = 0
		neigh_lab = nearest_neigh(k, test_point, train_xy)
		assert len(neigh_lab) == k
		for _, lab in neigh_lab: 
			assert lab != -1
			lab_sum += lab
		y_pred = 1 if lab_sum > k/2.0 else 0
		error_01 += 1 if y_pred != label else 0
	return float(error_01)/len(test_x)


def main():
	train_data_size = 200
	test_data_size = 100
	list_error_1NN = []
	list_error_3NN = []

	list_dim = range(11, 100, 10)
	# list_dim = [1, 11]
	for dim in list_dim:
		train_x, train_y = generate_data(dim, train_data_size)
		test_x, test_y = generate_data(dim, test_data_size)

		list_error = []
		KNN = 1
		list_error_1NN.append(do_kNN(KNN, test_x, test_y, train_x, train_y))
		KNN = 3
		list_error_3NN.append(do_kNN(KNN, test_x, test_y, train_x, train_y))
		
	print list_error_1NN
	print list_error_3NN
	plt.plot(list_dim, list_error_1NN, 'ro', label='1NN')
	plt.plot(list_dim, list_error_3NN, 'bo', label='3NN')
	plt.xlabel('Dimension')
	plt.ylabel('Error')
	plt.ylim(0,1)
	leg = plt.legend(loc='upper right', ncol=1, shadow=True, fancybox=True)
	leg.get_frame().set_alpha(0.5)
	plt.show()
	# plt.savefig("C:\\Users\\vedan\\Desktop\\Classes\\ML\\KNN_output.png")
	plt.close()

	# import pdb
	# pdb.set_trace()


if __name__ == "__main__":
	main()