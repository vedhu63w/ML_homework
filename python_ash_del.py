from scipy.io import loadmat
import numpy as np
import time

def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(np.linalg.norm(x1 - x2)) ** 2 / sigma ** 2)

def laplace_kernel(x1, x2, sigma):
    return np.exp(-(np.linalg.norm(x1 - x2)) / sigma)

def kernel_function(x1, x2, sigma, ktype='g'):
    if ktype == 'g':
        return gaussian_kernel(x1, x2, sigma)
    else:
        return laplace_kernel(x1, x2, sigma)

def generate_kernel_matrix(X, sigma, ktype='g'):
    n = X.shape[0]
    matrix = np.zeros((n,n), dtype=np.float)

    for i in range(n):
        for j in range(n):
            xi = X[i]
            xj = X[j]
            matrix[i,j] = kernel_function(xi, xj, sigma, ktype)
    return matrix

def get_alpha_and_matrix(X, Y, lambd, sigma, ktype='g'):
    n = X.shape[0]
    # print "Generating the matrix"
    start = time.time()
    K = generate_kernel_matrix(X, sigma, ktype)
    # print "Time taken = {}s".format(time.time() - start)
    # print "Calculating Inverse"
    print K
    start = time.time() 
    inv = np.linalg.inv(K + lambd * np.identity(n))
    # print "Time taken = {}s".format(time.time() - start)
    return inv.dot(Y), K

train_data = "79.mat"
test_data = "test79.mat"

x_train = loadmat(train_data)['d79'] / 255.0
def convert_fourier(X):
    X_fft = list()
    for i in range(X.shape[0]):
        x = X[i].reshape(28,28)
        # x_fft = np.fft.fft2(x).reshape(784)
        x_fft = np.fft.fft2(x)[:8,:8].reshape(64)
        X_fft.append(x_fft)
    return np.array(X_fft)
x_train = convert_fourier(x_train)        # Comment this if you do not want fourier transform

# 0 is for class '7'
# 1 is for class '9'
y_train = np.array([0.0 if i < 1000 else 1.0 for i in range(2000)])

print "Train data size:", x_train.shape, ",", y_train.shape

x_test = loadmat(test_data)['d79'] / 255.0
x_test = convert_fourier(x_test)        # Comment this if you do not want fourier transform
y_test = y_train

print "Test data size:", x_test.shape, ",", y_test.shape

print "Getting the model alpha"
count = 0.0
ktypes = ['g', 'l']
lambdas = [1e-8, 1e-2, 1e-1, 1.0, 10.0, 100.0]
# sigmas = [1e-1, 1.0, 10.0, 100.0]
sigmas = [ 100.0]
def classify(x, X, alpha, sigma, ktype='g'):
    n = X.shape[0]
    k_x = np.zeros(n, dtype=np.float)
    for i in range(n):
        k_x[i] = kernel_function(x, X[i], sigma, ktype)
    return k_x.dot(alpha)

for ktype in ktypes:
    for lambd in lambdas:
        for sigma in sigmas:
            count = 0.0
            alpha, K = get_alpha_and_matrix(x_train, y_train, lambd, sigma, ktype)
            print alpha
            # print alpha.dot(K.dot(alpha)) * lambd
            # print np.linalg.norm((K.dot(alpha)) - y_train)

            n_test = x_test.shape[0]

            for i in range(n_test):
                prediction = classify(x_test[i], x_train, alpha, sigma, ktype)
                # print prediction
                prediction = 0.0 if prediction < 0.5 else 1.0
                if prediction == y_test[i]:
                    count += 1.0

            print "Accuracy ktype = {} lambda = {} sigma = {} = {}".format(ktype, lambd, sigma, count / float(n_test))
