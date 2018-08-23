import numpy as np
import matplotlib.pyplot as plt

class LinearModel:
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate
    self.w1 = np.random.uniform()
    self.w0 = np.random.uniform()
    self.losses = []
    self.epochs = []

  def train(self, x, y, epochs):
    count = 0
    while(epochs>0):
      W1_array = np.full(x.shape, self.w1) #To create same valued w's for the batch
      W0_array = np.full(x.shape, self.w0)
      pred = W1_array*x + W0_array
      loss_w0 = (pred-y)
      square_loss = np.square(loss_w0)
      loss_w1 = loss_w0*x
      avg_loss_w1 = np.sum(loss_w1)/x.size
      avg_loss_w0 = np.sum(loss_w0)/x.size
      mean_square_loss = np.sum(square_loss)/x.size
      self.w1 = self.w1 - self.learning_rate*avg_loss_w1
      self.w0 = self.w0 - self.learning_rate*avg_loss_w0
      epochs = epochs - 1   
      count = count + 1
      self.epochs.append(count)
      self.losses.append(mean_square_loss)

  def showGraph(self):
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(self.epochs, self.losses)
    plt.title('Loss Curve')
    plt.show()

  def test(self, x, y):
    W1_array = np.full(x.shape, self.w1)
    W0_array = np.full(x.shape, self.w0)
    pred = W1_array*x + W0_array
    plt.ylabel('Output')
    plt.xlabel('Input')
    plt.plot(x, pred)
    plt.scatter(x, y)
    plt.title('Line fitting')
    plt.show()

def Main():
  dataset = np.load('linRegData.npy')
  x = dataset[:,0]
  y = dataset[:,1]
  myModel = LinearModel(0.001)
  myModel.train(x,y,100)
  myModel.showGraph()
  myModel.test(x,y)

if __name__ == '__main__':
  Main()
