import numpy as np
from sklearn import linear_model, tree, datasets
import matplotlib.pyplot as plt

n_samples = 100
#np.linspace(start,end,samples) generates evenly spaced values(samples) between start and end value
x = np.linspace(-np.pi,np.pi,n_samples)
#np.random.random(shape) generates values(shape) in the range [0,1) 
y = x/2 + np.sin(x) + np.random.random(x.shape)
#np.random.permutation(range) generates values within the specified range in arbitary order
random_indices = np.random.permutation(n_samples)

#training set 70%
x_train = x[random_indices[:70]]
y_train = y[random_indices[:70]]
#validation set 15%
x_valid = x[random_indices[70:85]]
y_valid = y[random_indices[70:85]]
#training set 70%
x_test = x[random_indices[85:]]
y_test = y[random_indices[85:]]

#linear model 
model = linear_model.LinearRegression()
#sklearn takes the inputs as matrices. Hence we reshpae the arrays into 2-D arrays
x_train_reshaped = np.matrix(x_train.reshape(len(x_train),1))
y_train_reshaped = np.matrix(y_train.reshape(len(y_train),1))
model.fit(x_train_reshaped, y_train_reshaped)

#evaluating and testing the model
mean_val_error = np.mean(np.square(y_valid - model.predict(np.matrix(x_valid.reshape(len(x_valid),1)))))
mean_test_error = np.mean(np.square(y_test - model.predict(np.matrix(x_test.reshape(len(x_test),1)))))
print 'Validation MSE: ', mean_val_error, '\nTest MSE: ', mean_test_error
plt.scatter(x,y,c='black')
plt.plot(x_train_reshaped, model.predict(x_train_reshaped), c='blue')
plt.xlabel('Input_Feature')
plt.ylabel('Output')
plt.title('Fig1: Mapping')
plt.show()
