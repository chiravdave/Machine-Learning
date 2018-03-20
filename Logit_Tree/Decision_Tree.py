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

#DecisionTree Classifier  
model = tree.DecisionTreeClassifier()
model.fit(X_train,Y_train)

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
print 'Test Classification Percentage =', correct_testing_percentage, '%'
