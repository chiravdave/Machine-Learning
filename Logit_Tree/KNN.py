import numpy as np
from sklearn import neighbors
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:,:2] #Choosing only the first two input-features
Y = iris.target
n_samples = len(Y)

#np.random.permutation(range) generates values within the specified range in arbitary order
random_indices = np.random.permutation(n_samples)

#training set 70%
X_train = X[random_indices[:70]]
Y_train = Y[random_indices[:70]]
#validation set 15%
X_valid = X[random_indices[70:85]]
Y_valid = Y[random_indices[70:85]]
#training set 70%
X_test = X[random_indices[85:]]
Y_test = Y[random_indices[85:]]

#Visualizing the training data, picking all the three classes
X_class0 = np.asmatrix([X_train[i] for i in range(len(X_train)) if Y_train[i]==0])
Y_class0 = np.zeros((X_class0.shape[0]),dtype=np.int)
X_class1 = np.asmatrix([X_train[i] for i in range(len(X_train)) if Y_train[i]==1])
Y_class1 = np.ones((X_class1.shape[0]),dtype=np.int)
X_class2 = np.asmatrix([X_train[i] for i in range(len(X_train)) if Y_train[i]==2])
Y_class2 = np.full((X_class2.shape[0]),fill_value=2,dtype=np.int)

#KNN Model
model = neighbors.KNeighborsClassifier(n_neighbors = 8)
model.fit(X_train, Y_train)

#validating and testing the model
validation = [model.predict(X_valid[i].reshape(1,len(X_valid[i])))[0] for i in range(X_valid.shape[0])]
correct_valid = 0
for i in range(len(validation)):
	if validation[i] == Y_valid[i]:
		correct_valid+=1
valid_percentage = (100*correct_valid)/len(Y_valid)
print 'Validation Percentage', valid_percentage, '%'

testing = [model.predict(X_test[i].reshape(1,len(X_test[i])))[0] for i in range(X_test.shape[0])]
correct_test = 0
for i in range(len(testing)):
	if testing[i] == Y_test[i]:
		correct_test+=1
test_percentage = (100*correct_test)/len(Y_test)
print 'Testing Percentage', test_percentage, '%'

plt.scatter(np.array(X_class0[:,0]).flatten(), np.array(X_class0[:,1]).flatten(),color='red')
plt.scatter(np.array(X_class1[:,0]).flatten(), np.array(X_class1[:,1]).flatten(),color='blue')
plt.scatter(np.array(X_class2[:,0]).flatten(), np.array(X_class2[:,1]).flatten(),color='green')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(['class 0','class 1','class 2'])
plt.title('Fig 3: Visualization of training data')
plt.show()
