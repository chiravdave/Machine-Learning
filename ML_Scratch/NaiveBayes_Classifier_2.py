import glob
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

words = []
labels = []
string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
special_characters = ['br','--','<<','>>',' ']
stopWords = set(stopwords.words('english'))
ps = PorterStemmer()

def prepareData():
    global words, labels
    curr = os.getcwd()
    #Reading data from negative folder
    for filename in glob.glob(os.path.join(curr,'movie/neg/*.txt')):
        f_read = open(filename, 'r')
        words.append(f_read.read())
        labels.append(0)
    #Reading data from positive folder
    for filename in glob.glob(os.path.join(curr,'movie/pos/*.txt')):
        f_read = open(filename, 'r')
        words.append(f_read.read())
        labels.append(1)

class NaiyeBayes:
    def __init__(self, split_ratio):
        self.split_ratio = split_ratio

    def removeStopWords(self, frequency, data, flag):
        global stopWords, ps, special_characters
        if(flag == True):
            for i in data:
                i = i.lower()
                words = word_tokenize(i)
                for word in words:
                    word = word.strip(string.punctuation)
                    word = ps.stem(word)
                    if word not in stopWords and word not in special_characters:
                        if word not in frequency:
                            frequency[word] = 1
                        else:
                            frequency[word] = frequency[word] + 1
        else:
            dense_test_words = []
            for i in data:
                new_data = []
                i = i.lower()
                words = word_tokenize(i)
                for word in words:
                    word = word.strip(string.punctuation)
                    word = ps.stem(word)
                    if word not in stopWords and word not in special_characters:
                        new_data.append(word)
                        if word not in frequency:
                            frequency[word] = 1
                        else:
                            frequency[word] = frequency[word] + 1
                dense_test_words.append(new_data)
            return dense_test_words

    def train(self):
        global words, labels
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(words, labels, test_size=self.split_ratio)
        neg_samples = []
        pos_samples = []
        self.total_pos = 0
        self.total_neg = 0
        #Separating positive and negative samples from the training dataset
        for i in range(len(self.y_train)):
            if self.y_train[i] == 1:
                self.total_pos = self.total_pos + 1
                pos_samples.append(self.x_train[i])
            else:
                self.total_neg = self.total_neg + 1
                neg_samples.append(self.x_train[i])
        #Calculating priors
        self.prior_pos = self.total_pos / (self.total_pos + self.total_neg)
        self.prior_neg = 1 - self.prior_pos
        self.freq_words_in_pos  = {} 
        self.removeStopWords(self.freq_words_in_pos, pos_samples, True)
        self.freq_words_in_neg  = {}
        self.removeStopWords(self.freq_words_in_neg, neg_samples, True)

    def test(self):
        frequency = {}
        dense_test_words = self.removeStopWords(frequency, self.x_test, False)
        length_test_data = len(self.y_test)
        vocab_set = len(frequency) 
        correct_pred = 0
        #Calculating posterior probabilities
        for i in range(length_test_data):
            posterior_pos_prob = self.prior_pos
            posterior_neg_prob = self.prior_neg
            for word in dense_test_words[i]:
                if word in self.freq_words_in_pos:
                    posterior_pos_prob = (posterior_pos_prob * self.freq_words_in_pos[word])/ self.total_pos
                else:
                    posterior_pos_prob = (posterior_pos_prob * (frequency[word] + 1))/ (length_test_data + vocab_set)
                if word in self.freq_words_in_neg:
                    posterior_neg_prob = (posterior_neg_prob * self.freq_words_in_neg[word])/ self.total_neg
                else:
                    posterior_neg_prob = (posterior_neg_prob * (frequency[word] + 1))/ (length_test_data + vocab_set)
            if(posterior_pos_prob > posterior_neg_prob):
                predicted_label = 1
            else:
                predicted_label = 0
            if(predicted_label == self.y_test[i]):
                correct_pred = correct_pred + 1 
        return (correct_pred/length_test_data)

    def trainSize(self):
        return len(self.x_train)

if __name__ == '__main__':
    prepareData()
    train_size = []
    accuracy = []
    split_ratios = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    for i in split_ratios:
        classifier = NaiyeBayes(i)
        accuracies = []
        for i in range(5):
            print('Fold : {}'.format(i+1))
            classifier.train()
            accuracies.append(classifier.test())
        accuracy.append(np.mean(accuracies))
        train_size.append(classifier.trainSize())
    plt.scatter(train_size, accuracy)
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy VS Training')
    plt.show()
