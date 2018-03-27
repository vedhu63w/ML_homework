import scipy
import scipy.io as sio

from   numpy						import array, dot, eye, square
from   numpy.linalg					import inv, norm
from   scipy.spatial.distance 		import pdist, squareform

def Gaussian_kernel(X, s):
	pairwise_dists = squareform(pdist(X, 'euclidean'))
	return scipy.exp(-pairwise_dists ** 2 / s ** 2)

# list_lambd = [100, 10, 1, 0.1, 0.01]
# list_s = [0.1, 1, 10, 100, 500]
list_lambd = [10e-8]
list_s = [100]
data = sio.loadmat("79.mat")
train_data = data['d79']
train_data = train_data / 255.0

test_file = sio.loadmat("test79.mat")
test_data = test_file['d79']
test_data = test_data / 255.0

for s in list_s:
	for lambd in list_lambd:
		K = Gaussian_kernel(train_data, s)
		Y = array([0 if i < 1000 else 1 for i in range(2000)])
		N = len(K)
		alpha = dot(inv(K + lambd * eye(N)), Y)
		print "Minimizer %f" % (sum(square(dot(alpha, K) - Y)) + lambd*dot(alpha, dot(K, alpha)))
		Y_test = array([0 if i < 1000 else 1 for i in range(2000)])
		loss = 0 
		for i in range(len(test_data)):
			a_j = 0
			for j in range(len(data['d79'])):
				norm_sq = square(norm(test_data[i] - train_data[j]))	
				# print norm_sq
				a_j += alpha[j]*scipy.exp(-norm_sq / s**2)
			y_pred = 1 if a_j > 0.5 else 0
			loss += 1 if y_pred!=Y_test[i] else 0
		print "lambda %f and s %f" % (lambd, s)
		print "Accuracy %f " % (1.0 - loss/float(len(test_data)))
