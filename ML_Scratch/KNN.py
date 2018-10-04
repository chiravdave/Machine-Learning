import numpy as np
import matplotlib.pyplot as plt
from load_dataset import read

class KNN():

    def train(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY

    def test(self, k, testX, testY):
        accuracy = 0
        length = testX.shape[0]
        for i in range(length):
            test_sample = testX[i]
            square_difference = np.square(self.trainX - test_sample)
            euclidean = np.sqrt(np.sum(square_difference, axis=1))
            indexes = np.argsort(euclidean)
            class_label = self.checkLabel(k, indexes)
            if class_label == testY[i]:
                accuracy+=1
        return accuracy/length

    def checkLabel(self, k, indexes):
        votes = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
        votes[self.trainY[indexes[0]]] = votes[self.trainY[indexes[0]]] + 1
        for i in range(1,k):
            votes[self.trainY[indexes[i]]] = votes[self.trainY[indexes[i]]] + 1
        class_label = 0
        max_vote = votes[0]
        for key,value in votes.items():
            if value > max_vote:
                class_label = key
                max_vote = value
        return class_label 

def Main():
    trainY, trainX = read()
    testY, testX = read(dataset='testing')
    trainX = trainX.reshape(-1, 784)
    testX = testX.reshape(-1, 784)
    accuracies = []
    k_list = [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]
    model = KNN()
    model.train(trainX, trainY)
    for k in k_list:
        accuracy = model.test(k, testX, testY)
        print(k,":",accuracy)
        accuracies.append(accuracy)
    plt.scatter(k_list, accuracies)
    plt.plot(k_list, accuracies)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Testing')
    plt.show()

if __name__ == '__main__':
    Main()
