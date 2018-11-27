from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(7)

class GMM():

    def __init__(self, data):
        self.data = data
        #Initialize gaussian models
        self.models = { 'mean1':data[63], 'weight1':0.5, 'mean2':data[125], 'weight2':0.5} 

    def fit(self):
        #Covariance matrix for clusters
        cov = np.cov(self.data.T)
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)
        #Probability distribution of every data point over all clusters
        self.prob = np.zeros((self.data.shape[0],2))
        old_weight1 = 0
        while(abs(self.models['weight1'] - old_weight1)>0.01):
            #E-Step
            for j in range(self.data.shape[0]):
                self.prob[j][0] = (np.exp(-0.5*np.dot(np.dot((self.data[j] - self.models['mean1']),cov_inv),(self.data[j] - self.models['mean1']).T))/np.sqrt(2*np.pi*cov_det))*self.models['weight1']
                self.prob[j][1] = (np.exp(-0.5*np.dot(np.dot((self.data[j] - self.models['mean2']),cov_inv),(self.data[j] - self.models['mean2']).T))/np.sqrt(2*np.pi*cov_det))*self.models['weight2']
            self.prob = self.prob/np.sum(self.prob, axis=1, keepdims=True)
            #M-Step
            old_weight1 = self.models['weight1']
            self.models['weight1'] = 0
            self.models['mean1'] = np.zeros((1,self.data.shape[1]))
            self.models['weight2'] = 0
            self.models['mean2'] = np.zeros((1,self.data.shape[1]))
            expectation1 = 0
            expectation2 = 0
            for i in range(self.data.shape[0]):
                self.models['mean1'] = self.models['mean1'] + self.prob[i][0]*self.data[i]
                expectation1 = expectation1 + self.prob[i][0]
                self.models['mean2'] = self.models['mean2'] + self.prob[i][1]*self.data[i]
                expectation2 = expectation2 + self.prob[i][1] 	 
            self.models['mean1'] = self.models['mean1']/expectation1
            self.models['weight1'] = expectation1/self.data.shape[0]
            self.models['mean2'] = self.models['mean2']/expectation2
            self.models['weight2'] = expectation2/self.data.shape[0]

    def plotGMM(self):
        cluster1_x1 = []
        cluster1_x2 = []
        cluster2_x1 = []
        cluster2_x2 = []
        for i in range(self.data.shape[0]):
            if(self.prob[i][0]>= 0.5):
                cluster1_x1.append(self.data[i][0])
                cluster1_x2.append(self.data[i][1])
            else:
                cluster2_x1.append(self.data[i][0])
                cluster2_x2.append(self.data[i][1])
        plt.scatter(cluster1_x1, cluster1_x2, c='red')
        plt.scatter(cluster2_x1, cluster2_x2, c='green')
        plt.xlabel('Feature1')
        plt.ylabel('Feature2')
        plt.title('GMM')
        plt.show()
                
def Main():
    data = pd.read_csv("./audioData.csv")
    features = data.values
    model = GMM(features)
    model.fit()	
    model.plotGMM()

if __name__ == '__main__':
    Main()
