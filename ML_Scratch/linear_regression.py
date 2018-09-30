import numpy as np
import matplotlib.pyplot as plt

class LinearModel:
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate
    self.W1 = np.random.rand(1,1)     #Weight matrix
    self.W0 = np.random.rand(1,1)     #Bias

  def train(self, X, Y, iterations):
    samples = X.size
    count = 1
    losses = []
    epochs = []
    while(count <= iterations):
      prediction = np.dot(self.W1,X) + self.W0  # Y = W1*X + W0
      dprediction = prediction - Y              #dL/dY
      square_loss = np.sum(np.square(dprediction))
      dW1 = np.dot(dprediction, X.T)/samples  #dL/dW1 = dL/dY*d(X.T)  
      dW0 = np.sum(dprediction, keepdims=True)/ samples              #dL/dW0 = dL/dY
      self.W1 = self.W1 - self.learning_rate * dW1
      self.W0 = self.W0 - self.learning_rate * dW0
      count = count + 1
      epochs.append(count)
      losses.append(square_loss)
    plt.scatter(epochs, losses)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Training')
    plt.show()

  def test(self, X, Y):
    prediction = np.dot(self.W1,X) + self.W0
    plt.plot(X.flatten(), Y.flatten())
    plt.plot(X.flatten(), prediction.flatten())
    plt.ylabel('Prediction')
    plt.xlabel('Input')
    plt.title('Testing')
    plt.show()

def Main():
  X = np.linspace(-np.pi,np.pi,200).reshape(-1,200)
  Y = np.sin(X)     
  myModel = LinearModel(0.001)
  myModel.train(X,Y,40)
  myModel.test(X,Y)

if __name__ == '__main__':
  Main()
