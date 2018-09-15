from __future__ import division, print_function
import numpy as np
from sklearn import datasets, svm 
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

#Loading iris dataset
iris = datasets.load_iris()
X = iris.data[:,:2]
Y = iris.target

# Splitting data into training set (75%) and testing set (25%) 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

def evaluate_on_test_data(model=None):
	if (model != None):
		predictions = model.predict(X_test)
		correct_predictions = 0
		for i in range(len(predictions)):
			if predictions[i] == Y_test[i]:
				correct_predictions += 1
		correct_percentage = (100*correct_predictions)/len(predictions)
		return correct_percentage
	else:
		return None

kernels = ['linear','poly','rbf']
accuracies = []
for _,kernel in enumerate(kernels):
	model = svm.SVC(kernel=kernel)
	model.fit(X_train, Y_train)
	acc = evaluate_on_test_data(model)
	accuracies.append(acc)
	print("{} % accuracy obtained with kernel = {}".format(acc, kernel))
