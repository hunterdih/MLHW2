import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy.matlib
from scipy.stats import multivariate_normal
# Reference for the MAP and MLE models are given to Luckecaino Melo for his Python Library creation
# https://medium.com/@luckecianomelo/the-ultimate-guide-for-linear-regression-theory-918fe1acb380
from mpl_toolkits.mplot3d import Axes3D

random.seed(20)
N = 100
x_dim = 2
sigma = 2


def train_MLE(train_x, train_y):
    w = None
    bias = np.ones((train_x.shape[0], 1))
    X = np.concatenate((train_x, bias), axis=1)
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(train_y)
    return w


def predict_MLE(w, test_x):
    bias = np.ones((test_x.shape[0], 1))
    X = np.concatenate((test_x, bias), axis=1)
    return X.dot(w)


def train_MAP(train_x, train_y, sigma=0.01, gamma=1):
    train_x = np.asarray(train_x).T
    train_y = np.asarray(train_y).T
    bias = np.ones((train_x.shape[0], 1))
    X = np.concatenate((train_x, bias), axis=1)
    # Calcualte gammas

    A = (X.T.dot(X) + (sigma / gamma) * np.eye(X.shape[1]))
    B = X.T.dot(y_train)
    W = np.linalg.inv(A).dot(B)

    return W


def predict_MAP(test_x, w):
    test_x = np.asarray(test_x).T
    bias = np.ones((test_x.shape[0], 1))
    X = np.concatenate((test_x, bias), axis=1)
    return X.dot(w)


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
    gmmParameters['covMatrices'][:, :, 1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
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
    fig7 = plt.figure(7)
    ax = fig7.add_subplot(projection='3d')
    ax.scatter(a, b, c, marker=mark, color=col)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title('Training Dataset')


if __name__ == '__main__':
    x_train, y_train, x_validate, y_validate = hw2q2()

    wML = train_MLE(x_train.T, y_train.T)
    resultsML = predict_MLE(wML, x_validate.T)

    gamma_list = list(np.logspace(-10, 10, 10000))
    MSEML = np.mean(np.fabs(y_validate ** 2 - resultsML ** 2))
    MSEMAP_list = []
    for g in gamma_list:
        wMAP = train_MAP(x_train, y_train, gamma=g)
        resultsMAP = predict_MAP(x_validate, wMAP)
        MSEMAP_list.append(np.mean(np.fabs(y_validate ** 2 - resultsMAP ** 2)))

    msemap_dataframe = pd.DataFrame(MSEMAP_list).describe()
    fig0 = plt.figure(0)
    plt.plot(gamma_list, MSEMAP_list, label='MAP Classification MSE')
    plt.axhline(y=MSEML, color='r', label='ML Classification MSE')
    plt.xscale('log')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Gamma')
    plt.title('MAP and ML Classification Mean Squared Error vs. Gamma')
    plt.legend()
    plt.grid()

    trainResultsML = predict_MLE(wML, x_train.T)
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(projection='3d')

    ax.scatter(x_train[0], x_train[1], trainResultsML, marker='+', color='g', label='ML Estimation')
    ax.scatter(x_train[0], x_train[1], y_train, marker='o', color='b', label='Ground Truth')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title("ML Estimation Vs. Ground Truth (Training)")
    plt.legend()

    fig2 = plt.figure(2)
    ax = fig2.add_subplot(projection='3d')
    ax.scatter(x_validate[0], x_validate[1], resultsML, marker='+', color='g', label='ML Estimation')
    ax.scatter(x_validate[0], x_validate[1], y_validate, marker='o', color='b', label='Ground Truth')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title("ML Estimation Vs. Ground Truth (Validation)")
    plt.legend()

    gamma = gamma_list[MSEMAP_list.index(min(MSEMAP_list))]
    wMAP = train_MAP(x_train, y_train, gamma=gamma)
    trainResultsMAP = predict_MAP(x_train, wMAP)
    validationResultsMAP = predict_MAP(x_validate, wMAP)

    fig3 = plt.figure(3)
    ax = fig3.add_subplot(projection='3d')

    ax.scatter(x_train[0], x_train[1], trainResultsMAP, marker='+', color='g', label='MAP Estimation')
    ax.scatter(x_train[0], x_train[1], y_train, marker='o', color='b', label='Ground Truth')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title("MAP Estimation Vs. Ground Truth (Training)")
    plt.legend()

    fig4 = plt.figure(4)
    ax = fig4.add_subplot(projection='3d')
    ax.scatter(x_validate[0], x_validate[1], validationResultsMAP, marker='+', color='g', label='MAP Estimation')
    ax.scatter(x_validate[0], x_validate[1], y_validate, marker='o', color='b', label='Ground Truth')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title("MAP Estimation Vs. Ground Truth (Validation)")
    plt.legend()

    gamma = gamma_list[0]
    wMAP = train_MAP(x_train, y_train, gamma=gamma)
    trainResultsMAP = predict_MAP(x_train, wMAP)
    validationResultsMAP = predict_MAP(x_validate, wMAP)

    fig5 = plt.figure(5)
    ax = fig5.add_subplot(projection='3d')
    ax.scatter(x_validate[0], x_validate[1], validationResultsMAP, marker='+', color='g', label='MAP Estimation')
    ax.scatter(x_validate[0], x_validate[1], y_validate, marker='o', color='b', label='Ground Truth')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title("MAP Estimation Vs. Ground Truth (Validation, g = 10^-10)")
    plt.legend()
    plt.show()
    print(f'{MSEML=}')
