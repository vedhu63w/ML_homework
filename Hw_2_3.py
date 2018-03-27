import scipy
import scipy.io as sio

from   numpy						import array, dot, exp, eye, fft, square, zeros
from   numpy.linalg					import inv, norm
from   scipy.spatial.distance 		import pdist, squareform


def Gaussian_kernel(X, s):
	pairwise_dists = squareform(pdist(X, 'euclidean'))
	return scipy.exp(-pairwise_dists ** 2 / s ** 2)

def gaussian_kernel(x1, x2, s):
    return exp(-(norm(x1 - x2)) ** 2 / s ** 2)

def generate_kernel_matrix(X, s):
    n = X.shape[0]
    matrix = zeros((n,n), dtype=float)

    for i in range(n):
        for j in range(n):
            xi, xj = X[i], X[j]
            matrix[i,j] = gaussian_kernel(xi, xj, s)
    return matrix

def get_K(X, Y, lambd, s):
    n = X.shape[0]
    K = generate_kernel_matrix(X, s)
    return K

def convert_fourier(X):
    X_new = list()
    for i in range(X.shape[0]):
        x = X[i].reshape(28,28)
        # x_fft = fft.fft2(x).reshape(784)
        x_fft = fft.fft2(x)[:8,:8].reshape(64)
        X_new.append(x_fft)
    return array(X_new)


# list_lambd = [100, 10, 1, 0.1, 0.01]
# list_s = [0.1, 1, 10, 100, 500]
list_lambd = [1e-8]
list_s = [100.0]
data = sio.loadmat("79.mat")
train_data = data['d79'] / 255.0
train_data = convert_fourier( train_data)

test_file = sio.loadmat("test79.mat")
test_data = test_file['d79'] / 255.0
test_data = convert_fourier( test_data)

for s in list_s:
	for lambd in list_lambd:
		# Something is not working in this one
		# K = Gaussian_kernel(train_data, s)
		Y = array([0 if i < 1000 else 1 for i in range(2000)])
		K = get_K(train_data, Y, lambd, s)
		# print K
		N = len(K)
		alpha = dot(inv(K + lambd * eye(N)), Y)
		print "Minimizer %f" % (sum(square(dot(alpha, K) - Y)) + lambd*dot(alpha, dot(K, alpha)))
		Y_test = array([0 if i < 1000 else 1 for i in range(2000)])
		loss = 0 
		for i in range(len(test_data)):
			a_j = 0
			for j in range(len(train_data)):
				norm_sq = square(norm(test_data[i] - train_data[j]))	
				# print norm_sq
				a_j += alpha[j]*scipy.exp(-norm_sq / s**2)
			y_pred = 1 if a_j > 0.5 else 0
			loss += 1 if y_pred!=Y_test[i] else 0
		print "lambda %f and s %f" % (lambd, s)
		print "Accuracy %f " % (1.0 - loss/float(len(test_data)))
