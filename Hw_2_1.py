import scipy
import scipy.io as sio

from   numpy						import array, dot, eye, square
from   numpy.linalg					import inv, norm
from   sklearn 						import linear_model
from   scipy.spatial.distance 		import pdist, squareform
from   sklearn.svm 					import LinearSVC


data = sio.loadmat("79.mat")
train_data = data['d79']
train_data = train_data / 255.0
Y = array([0 if i < 1000 else 1 for i in range(2000)])

test_file = sio.loadmat("test79.mat")
test_data = test_file['d79']
test_data = test_data / 255.0
Y_test = array([0 if i < 1000 else 1 for i in range(2000)])

clf = LinearSVC(random_state=0)
clf.fit(train_data, Y)
Y_pred = clf.predict(test_data)
print "Accuracy SVM %f" % (1 - sum([1 for i in range(len(Y_pred)) if Y_pred[i]!=Y_test[i]])/float(len(Y_pred)))

regr = linear_model.LinearRegression()
regr.fit(train_data, Y)
Y_pred = regr.predict(test_data)
Y_pred = [1 if Y_pred[i] > 0.5 else 0 for i in range(len(Y_pred))]
print "Accuracy LR %f" % (1 - sum([1 for i in range(len(Y_pred)) if Y_pred[i]!=Y_test[i]])/float(len(Y_pred)))