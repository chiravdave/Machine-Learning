from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(7)

class GMM():

    def __init__(self, data):
        self.data = data
        #Initialize gaussian models
        self.models = { 'mean1':data[np.random.randint(127)], 'weight1':0.5, 'mean2':data[np.random.randint(127)], 'weight2':0.5}

    def fit(self):
        #Covariance matrix for clusters
        cov = np.cov(self.data.T)
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)
        #Probability distribution of every data point over all clusters
        self.prob = np.zeros((self.data.shape[0],2))
        old_weight = 0
        while(abs(self.models['weight1'] - old_weight) >= 0.001):
            #E-Step
            for j in range(self.data.shape[0]):
                data_prob1 = np.exp(-0.5*np.dot(np.dot((self.data[j] - self.models['mean1']),cov_inv),(self.data[j] - self.models['mean1']).T))/np.sqrt(((2*np.pi)**13)*cov_det)
                data_prob2 = np.exp(-0.5*np.dot(np.dot((self.data[j] - self.models['mean2']),cov_inv),(self.data[j] - self.models['mean2']).T))/np.sqrt(((2*np.pi)**13)*cov_det)
                data_prob_sum = data_prob1 + data_prob2
                self.prob[j][0] = data_prob2 / data_prob_sum
                self.prob[j][1] = 1 - self.prob[j][0]
            #M-Step
            old_weight = self.models['weight1']
            self.models['mean1'] = np.zeros((1,13))
            self.models['mean2'] = np.zeros((1,13))
            for i in range(self.data.shape[0]):
                self.models['mean1'] = self.models['mean1'] + self.prob[i][0]*self.data[i]
                self.models['mean2'] = self.models['mean2'] + self.prob[i][1]*self.data[i]
            expectation_sum = np.sum(self.prob, axis=0) 	 
            self.models['mean1'] = self.models['mean1']/expectation_sum[0]
            self.models['weight1'] = expectation_sum[0]/self.data.shape[0]
            self.models['mean2'] = self.models['mean2']/expectation_sum[1]
            self.models['weight2'] = 1 - self.models['weight1'] 

    def plotGMM(self):
        colors = []
        for i in range(self.data.shape[0]):
            if(self.prob[i][0]>= 0.5):
                colors.append('red')
            else:
                colors.append('green')
        plt.scatter(self.data[:,0],self.data[:,1], c=colors)
        plt.xlabel('Feature1')
        plt.ylabel('Feature2')
        plt.title('GMM')
        plt.show()
                
def Main():
    data = pd.read_csv("./audioData.csv")
    numpy_array = data.values
    model = GMM(numpy_array)
    model.fit()	
    model.plotGMM()

if __name__ == '__main__':
    Main()
