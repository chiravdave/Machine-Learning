import numpy as np
from sklearn import linear_model, tree, datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
#Choosing only the first two input-features
X = iris.data[:,:2] 
Y = iris.target
X = X[:100]
Y = Y[:100]
n_samples = len(Y)
random_indices = np.random.permutation(n_samples)
#training set 70%
n_training_samples = int(n_samples*0.7)
X_train = X[random_indices[:n_training_samples]]
Y_train = Y[random_indices[:n_training_samples]]
#validation set 15%
n_validation_samples = int(n_samples*0.15)
X_valid = X[random_indices[n_training_samples:n_training_samples + n_validation_samples]]
Y_valid = Y[random_indices[n_training_samples:n_training_samples + n_validation_samples]]
#training set 15%
n_testing_samples = int(n_samples*0.15)
X_test = X[random_indices[-n_testing_samples:]]
Y_test = Y[random_indices[-n_testing_samples:]]

#Visualizing the training data
X_class0 = np.asmatrix([X_train[i] for i in range(len(X_train)) if Y_train[i]==0]) #Picking only the first two classes
Y_class0 = np.zeros((X_class0.shape[0]),dtype=np.int)
X_class1 = np.asmatrix([X_train[i] for i in range(len(X_train)) if Y_train[i]==1])
Y_class1 = np.ones((X_class1.shape[0]),dtype=np.int)

#Logistic Regression 
model = linear_model.LinearRegression()
full_X = np.concatenate((X_class0,X_class1),axis=0)
full_Y = np.concatenate((Y_class0,Y_class1),axis=0)
model.fit(full_X,full_Y)

#(Visualization code taken from: http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html)
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
h = .02  # step size in the mesh
x_min, x_max = full_X[:, 0].min() - .5, full_X[:, 0].max() + .5
y_min, y_max = full_X[:, 1].min() - .5, full_X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) 

#predict for the entire mesh to find the regions for each class in the feature space
validation = [model.predict(X_valid[i].reshape((1,2)))[0] for i in range(X_valid.shape[0])]
validation_correct = 0
for i in range(len(validation)):
    if validation[i]==Y_valid[i]:
        validation_correct+= 1
correct_validation_percentage = (validation_correct/len(Y_valid))*100
print 'validation Classification Percentage =', correct_validation_percentage, '%'

testing = [model.predict(X_test[i].reshape((1,2)))[0] for i in range(X_test.shape[0])]
testing_correct = 0
for i in range(len(testing)):
    if testing[i]==Y_test[i]:
        testing_correct+= 1
correct_testing_percentage = 100*(testing_correct/len(Y_test))
print 'test misclassification percentage =', correct_testing_percentage, '%'

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(np.array(X_class0[:, 0]).flatten(), np.array(X_class0[:, 1]).flatten(), c='red', edgecolors='k', cmap=plt.cm.Paired)
plt.scatter(np.array(X_class1[:, 0]).flatten(), np.array(X_class1[:, 1]).flatten(), c='blue', edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Fig 4: Visualization of decision boundary')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
