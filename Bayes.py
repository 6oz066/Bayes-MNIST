from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt

# Gasussian Naive Bayes
class GS_bayes():
    def __init__(self):
        self.Gsmodel = None
        self.xtrain=np.array(x_train)
        self.ytrain=np.array(y_train)
        self.xtest=np.array(x_test)
        self.ytest=np.array(y_test)


    def gray_evaluation(self):
        i=self.xtrain[0]
        hist, bins = np.histogram(i, bins=256, range=[0, 256])
        plt.title('Grayscale Histogram')
        plt.xlabel('Grayscale Value')
        plt.ylabel('Pixel Count')
        plt.plot(bins[:-1], hist)
        plt.xlim([0, 256])
        plt.ylim([0, np.max(hist) * 0.5])
        plt.grid(True)
        plt.show()



    # Calculate Evaluation, variance and mean of gradient
    def get_character(self):
        # Variance
        var=[]
        for i in self.xtrain:
            var.append(np.mean(i))
        # Gradient
        grad=[]
        for i in self.xtrain:
            grad_x = cv2.Sobel(self.xtrain, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(self.xtrain, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = cv2.magnitude(grad_x, grad_y)
            grad.append(grad_mag)
        return var,grad

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
    gnb_time=gnb.train()
    gnb.predict()
    gnb.gray_evaluation()
    # print("The evaluation, variance and mean of gradient are ",gnb.get_character())
    print("The accuracy of bayes is",gnb.test_accuracy())
    print("The training time of bayes is",gnb_time)
