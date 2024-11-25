from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import time
import NeuMain as NM

# Gasussian Naive Bayes
class GS_bayes():
    def __init__(self):
        self.Gsmodel = None
        self.xtrain=x_train
        self.ytrain=y_train
        self.xtest=x_test
        self.ytest=y_test

    # Calculate Evaluation, variance and mean of gradient
    def get_character(self):


    def train(self):
        self.Gsmodel = GaussianNB()
        start = time.time()
        self.Gsmodel.fit(self.xtrain,self.ytrain)
        stop = time.time()
        execution_time = stop - start
        return execution_time

    def test_accuracy(self):
        acc = self.Gsmodel.score(self.xtest,self.ytest)
        return acc

    def predict(self):
        y_pred = self.Gsmodel.predict(self.xtest)
        y_prob = self.Gsmodel.predict_proba(self.xtest)
        return y_pred, y_prob

if __name__ == '__main__':
    # Load MNIST data
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    gnb = GS_bayes()
    print(gnb.train())
    gnb.predict()
    print(gnb.test_accuracy())