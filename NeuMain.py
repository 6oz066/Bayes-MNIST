from random import random
import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

def or_relu(x):
    return np.maximum(0, x)

def d_or_relu(x):
    return 1. * (x > 0)


def relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def d_relu(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


# Sigmoid Function
def sigmoid(x):
    return 1/(1+np.exp(-x))


# Derivative of sigmoid, use the parameter from f(x)
def d_sigmoid(f):
        return f*(1.0-f)


def tanh(x):
    return np.tanh(x)


def d_tanh(f):
    return 1-f**2


# Jump out of Vanishing Gradient
def jp_vg(martix):
    for i in martix:
        for j in i:
            if j==0:
                j+=random()
            elif j==1:
                j-=random()


# Judgement Function for output result
def judge_result(output,test):
    return (abs(output-test)<0.6)


# Input layer class
class Input_layer:
    def __init__(self,in_value):
        # input value
        self.value = in_value
        self.weight = np.random.rand((in_value).shape[1],1)

    def active(self,i,function):
        return function(np.sum(np.dot(self.value[i],self.weight)))

    def topreditct(self,x_test):
        self.value = x_test


class Hidden_layer:
    def __init__(self,i_line,i_col,learning):
        # Number of element per level
        self.num_line = i_line
        # Number of level
        self.num_col = i_col
        # Learning rate
        self.learning = learning

        # Build the structure
        # self.weight_martix = np.random.rand(self.num_col,self.num_line)*10
        self.weight_martix = np.random.randn(self.num_col, self.num_line) * np.sqrt(2 / self.num_line)
        self.threshold_martix = np.random.rand(self.num_col,self.num_line)
        self.bias = np.zeros((self.num_col,self.num_line))
        self.loss = np.zeros((self.num_col,self.num_line))

        # Output
        self.Output_value=np.zeros((self.num_col,self.num_line))

    # Tool Function - Do not use directly
    def output_function(self,input,function,index):
        return function(input-self.threshold_martix[index[0],index[1]])

    def batch_norm(self,x, gamma=1, beta=0, epsilon=1e-5):
        mean = np.mean(x)
        var = np.var(x)
        x_norm = (x - mean) / np.sqrt(var + epsilon)
        return gamma * x_norm + beta

    # For test data, do not use while training
    def forward_calculate(self,Input):
        for i in range(self.num_col):
            # Avoid Vanishing Gradient
            jp_vg(self.Output_value)
            for j in range(self.num_line):
                if(i==0):
                    index = list([i, j])
                    self.Output_value[i,j] = self.output_function(np.sum(Input*self.weight_martix[i]),relu,index)

                else:
                    index = list([i, j])
                    self.Output_value[i,j] = self.output_function(np.sum(np.dot(self.Output_value[i-1], self.weight_martix[i])),relu,index)

        return self.Output_value[-1]

    # For train data, which will import forward method automatically
    def backward_train(self,x_train,y_train,Output_layer):

        # Use the loss from output layer to calculate
        for i in reversed(range(self.num_col)):

            if i==self.num_col-1:
                hidden_bias = Output_layer.backward_learning(y_train)
                for j in range(self.num_col):
                    self.bias[i,j] = hidden_bias*self.weight_martix[i,j]*d_relu(self.Output_value[i,j])
                    self.weight_martix[i,j] -= self.bias[i,j] * self.learning
                    self.threshold_martix[i,j] -= self.bias[i,j] *self.learning
            else:
                self.bias[i] = self.bias_function(i,d_relu)
                self.weight_martix[i] -= self.bias[i] * self.learning
                self.threshold_martix[i] -= self.bias[i] * self.learning


    # Calculate bias except the last level
    def bias_function(self,i,function):
        return np.dot(function(self.Output_value[i]),self.loss_function(i))

    # Calculate loss except the last level
    def loss_function(self,i):
        return np.dot(self.bias[i-1],self.weight_martix[i])


class Output_layer:
    def __init__(self,num_element,learning):
        # Weight
        self.weight = np.random.rand(1,num_element)*10
        self.threshold = random()
        self.output = 0
        self.learning = learning
        self.bias = 0

    # Function for outputting calculation
    def output_result(self,input):
        self.output = np.sum(np.dot(input,self.weight.T))
        return relu(-1*self.output)

    # Tool Function - Not use directly
    def output_function(self,input,function):
        return function(input-(self.threshold))

    def backward_learning(self,y_train):
        self.bias = self.bias_function(y_train,d_relu)
        self.weight -= self.learning* self.bias
        self.threshold -= self.learning* self.bias

        return self.bias

    def bias_function(self,y_train,dfunction):
        loss = self.loss_function(y_train)
        d_f = dfunction(self.output)
        return loss*d_f

    # Loss Function
    def loss_function(self,y_train):
        E_k=(-1)*y_train*math.log(abs(self.output))
        return E_k

class NeuralNetwork:
    def __init__(self, learning_rate, hiddenlevel,hiddennum, x_train_m, y_train_m, x_test_m, y_test_m):
        self.input_layer = Input_layer(x_train_m)
        self.learning_rate = learning_rate
        self.hiddenlevel= hiddenlevel
        self.hidden_layer = Hidden_layer(hiddennum, self.hiddenlevel, self.learning_rate)
        self.output_layer = Output_layer(hiddennum, self.learning_rate)
        self.x_train = x_train_m
        self.y_train = y_train_m
        self.x_test = x_test_m
        self.y_test = y_test_m

        self.correct=0

        # TO EDIT WITH
        self.function =relu
        self.dfunction = d_relu

    def train(self):
        print("Training active")
        for i in tqdm(range(len(self.x_train)),desc="Training process"):
            self.output_layer.output_result(self.hidden_layer.forward_calculate(self.input_layer.active(i,self.function)))
            self.hidden_layer.backward_train(self.x_train[i],self.y_train[i],self.output_layer)

        print("Training finished for loop times:",len(self.x_train))


    def predict(self):
        self.input_layer.topreditct(self.x_test)
        self.correct=0
        dynamic_correct_rate =0
        dynamic_correct_point = np.zeros(len(self.x_test),dtype=float)
        for i in tqdm(range(len(self.x_test)),desc="Testing process"):
            predict_result = self.output_layer.output_result(self.hidden_layer.forward_calculate(self.input_layer.active(i,relu)))

            if judge_result(predict_result,self.y_test[i]):
                self.correct+=1
                dynamic_correct_rate = self.correct/(i+1)
                dynamic_correct_point[i]= dynamic_correct_rate

        plot_x = np.linspace(0,len(self.x_test),len(self.x_test))
        plot_y = dynamic_correct_point
        plt.plot(plot_x,plot_y)
        plt.legend(["Loop time","Correct Rate"])
        plt.title("Dynamic Predicted Correct Rate")
        plt.show()

if __name__ == '__main__':

    # Load MNIST data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Standardize
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # The shape of the array
    print(x_train.shape, y_train.shape)  # 训练集的形状
    print(x_test.shape, y_test.shape)   # 测试集的形状

    # Flat into 1-dimension array
    x_train_flat = x_train.reshape(-1, 28 * 28)  # -1 表示自动计算该维度的大小
    x_test_flat = x_test.reshape(-1, 28 * 28)

    # Shape of flat
    print("扁平化后的训练集图像形状:", x_train_flat.shape)
    print("扁平化后的测试集图像形状:", x_test_flat.shape)


    model = NeuralNetwork(0.01,3,6,x_train_flat,y_train/10,x_test_flat,y_test/10)
    model.train()
    model.predict()

    # For test usage
    # Learning rate
    #     Learning = 0.5
    # a=Hidden_layer(784,2,0.5)
    # b=Output_layer(784,0.5)
    # print(b.output_result(a.forward_calculate(x_train_flat[100,:])))
    # print(y_train[1])
    # print(a.weight_martix)
    # for i,j in zip(x_train_flat,y_train/10):
    #     a.backward_train(i,j,b)
    # print(a.weight_martix)