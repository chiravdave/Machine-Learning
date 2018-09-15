import numpy as np
import matplotlib.pyplot as plt

class LinearModel:
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate

  def train(self, x, y, epochs):
    self.losses = []
    self.epochs = []
    self.W1 = np.random.rand(1,1)     #Weight matrix
    self.W0 = np.random.rand(1,1)          #Bias
    count = 1
    while(count <= epochs):
      prediction = np.dot(self.W1,x) + self.W0  # Y = W1*X + W0
      loss = prediction - y
      square_loss = np.sum(np.square(loss))
      dprediction = np.sum(loss, keepdims=True)/prediction.size  #dL/dy
      dW1 = np.sum(dprediction *  x.T, keepdims=True) 
      dW0 = dprediction * np.ones((1,1))
      self.W1 = self.W1 - self.learning_rate * dW1
      self.W0 = self.W0 - self.learning_rate * dW0
      count = count + 1
      self.epochs.append(count)
      self.losses.append(square_loss)

  def showGraph(self):
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.scatter(self.epochs, self.losses)
    plt.title('Loss Curve')
    plt.show()

  def test(self, x, y):
    prediction = np.dot(self.W1,x) + self.W0
    plt.ylabel('Output')
    plt.xlabel('Input')
    plt.plot(x.flatten(), prediction.flatten())
    plt.scatter(x, y)
    plt.title('Line fitting')
    plt.show()

def Main():
  dataset = np.load('linRegData.npy')   #100 data points
  x, y = np.hsplit(dataset, 2)
  myModel = LinearModel(0.001)
  myModel.train(x.T,y.T,20)
  myModel.showGraph()
  myModel.test(x.T,y.T)

if __name__ == '__main__':
  Main()
