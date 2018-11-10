from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(7)

class KNN_Neighbors():

    def __init__(self, k, data):
        self.data = data
        self.k = k
        min_values = data.min(axis=0)
        max_values = data.max(axis=0)
        features = min_values.shape[0]
        #Initialize cluster points
        self.clusters = np.zeros((features, k))
        for i in range(features):
            self.clusters[i] = np.random.randint(min_values[i], max_values[i], size=(k,))
        self.clusters = self.clusters.T

    def fit(self):
        clusters_old = np.zeros((self.k, self.data.shape[1]))
        #Storing cluster group of individual data points  
        cluster_group = np.zeros(self.data.shape[0])
        distortion = 0
        while self.L2(self.clusters, clusters_old, True) != 0:
            #Assigning each data point to its closest cluster
            for i in range(self.data.shape[0]):
                l2 = self.L2(self.data[i], self.clusters)
                index = np.argmin(l2)
                distortion = distortion + l2[index]
                cluster_group[i] = index
            #Storing old cluster values
            clusters_old = deepcopy(self.clusters)
            #Updating cluster values
            for i in range(self.k):
                data_points = [self.data[j] for j in range(self.data.shape[0]) if cluster_group[j] == i]
                if(len(data_points)>0):
                    self.clusters[i] = np.mean(data_points, axis=0)
        return distortion 

    def L2(self, x1, x2, flag=False):
        square_diff = np.square(x1-x2)
        if flag == True:
            return (np.sum(square_diff))
        else:
            return (np.sum(square_diff, axis=1))

def Main():
    data = pd.read_csv("./audioData.csv")
    numpy_array = data.values
    distortion = []
    k_list = []
    for k in range(2,11):
        model = KNN_Neighbors(k, numpy_array)
        cost = model.fit()	
        distortion.append(cost)
        k_list.append(k)
    plt.plot(k_list, distortion)
    plt.xlabel('K')
    plt.ylabel('Distortion')
    plt.title('KMeans')
    plt.show()

if __name__ == '__main__':
    Main()
