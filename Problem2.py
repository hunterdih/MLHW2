import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# Reference for the MAP and MLE models are given to Luckecaino Melo for his Python Library creation
# https://medium.com/@luckecianomelo/the-ultimate-guide-for-linear-regression-theory-918fe1acb380
from mpl_toolkits.mplot3d import Axes3D

random.seed(20)
N = 100
x_dim = 2
sigma = 2


def train_MLE(train_x, train_y):
    weights = None
    bias = np.ones((train_x.shape[0], 1))
    X = np.concatenate((train_x, bias), axis=1)
    weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(train_y)
    return weights


def predict_MLE(w, test_x):
    bias = np.ones((test_x.shape[0], 1))
    X = np.concatenate((test_x, bias), axis=1)
    return X.dot(w)


    def train(train_x, train_y, lambd):
        weights = None
        bias = np.ones((train_x.shape[0], 1))
        X = np.concatenate((train_x, bias), axis=1)
        weights = np.linalg.inv(X.T.dot(X) + lambd * np.eye(X.shape[1])).dot(X.T).dot(train_y)

    def predict(self, test_x):
        bias = np.ones((test_x.shape[0], 1))
        X = np.concatenate((test_x, bias), axis=1)
        return X.dot(self.weights)

def hw2q2():
    Ntrain = 100
    data = generateData(Ntrain)
    plot3(data[0, :], data[1, :], data[2, :])
    xTrain = data[0:2, :]
    yTrain = data[2, :]

    Ntrain = 1000
    data = generateData(Ntrain)
    plot3(data[0, :], data[1, :], data[2, :])
    xValidate = data[0:2, :]
    yValidate = data[2, :]

    return xTrain, yTrain, xValidate, yValidate


def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3, .4, .3]  # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:, :, 0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:, :, 1] = np.array([[8, 0, 0], [0, .5, 0], [0,  0, .5]])
    gmmParameters['covMatrices'][:, :, 2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    x, labels = generateDataFromGMM(N, gmmParameters)
    return x


def generateDataFromGMM(N, gmmParameters):
    #    Generates N vector samples from the specified mixture of Gaussians
    #    Returns samples and their component labels
    #    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors']  # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]  # Data dimensionality
    C = len(priors)  # Number of components
    x = np.zeros((n, N))
    labels = np.zeros((1, N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1, N))
    thresholds = np.zeros((1, C + 1))
    thresholds[:, 0:C] = np.cumsum(priors)
    thresholds[:, C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:, l]))
        Nl = len(indl[1])
        labels[indl] = (l + 1) * 1
        u[indl] = 1.1
        x[:, indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:, l], covMatrices[:, :, l], Nl))

    return x, labels


def plot3(a, b, c, mark="o", col="b"):

    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(a, b, c, marker=mark, color=col)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title('Training Dataset')



if __name__ == '__main__':
    x_train, y_train, x_validate, y_validate = hw2q2()

    weights = train_MLE(x_train.T, y_train.T)
    results = predict_MLE(weights, x_validate.T)

    mse = np.mean(np.fabs(y_validate**2 - results**2))

    print(f'{mse=}')


