#   Library Imports
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import random

#   Declare Constants

M01 = np.asarray(([-1, -1]))
M02 = np.asarray(([1, 1]))
M11 = np.asarray(([-1, 1]))
M12 = np.asarray(([1, -1]))
CIJ = np.asarray(([1, 0], [0, 1]))
W01 = 0.5
W02 = 0.5
W11 = 0.5
W12 = 0.5
#   Class priors
PL0 = 0.6
PL1 = 0.4
THEORETICAL_GAMMA = PL0 / PL1


def get_discriminant(data):
    # Calculate all possible gamma values
    gamma_list = []
    use_data = data.values[:, :2]

    # discs = np.log(W11 * multivariate_normal.pdf(use_data, M11, CIJ) + W12 * multivariate_normal.pdf(use_data, M12, CIJ)) - np.log(W01 * multivariate_normal.pdf(use_data, M01, CIJ) + W02 * multivariate_normal.pdf(use_data, M02, CIJ))
    # discs = np.log(multivariate_normal.pdf(use_data, M11, CIJ)) - np.log(multivariate_normal.pdf(use_data, M01, CIJ))

    LogVal1 = np.log((W11 * multivariate_normal.pdf(use_data, M11, CIJ)) + (W12 * multivariate_normal.pdf(use_data, M12, CIJ)))
    LogVal0 = np.log((W01 * multivariate_normal.pdf(use_data, M01, CIJ)) + (W02 * multivariate_normal.pdf(use_data, M02, CIJ)))
    discs = LogVal1 - LogVal0

    return discs


def erm_classify(discs):
    use_discs = np.asarray(discs)
    samples = len(list(discs))
    disc_0 = discs[:int(PL0 * samples)]
    disc_1 = discs[int(PL0 * samples):]

    tpp = []
    fpp = []
    err = []

    gammas_list = []
    use_discs = np.concatenate((use_discs, [-5]))
    use_discs = np.concatenate((use_discs, [5]))
    for g in np.sort(use_discs):
        # P(D = 1|L = 1)
        tpc = ((disc_1 >= g).sum() / (PL1 * samples))
        # P(D = 1|L = 0)
        fpc = ((disc_0 >= g).sum() / (PL0 * samples))

        tpp.append(tpc)
        fpp.append(fpc)

        # (error; γ) = P(D = 1|L = 0; γ)P(L = 0) + P(D = 0|L = 1; γ)P(L = 1)
        err.append((fpc * PL0) + ((1 - tpc) * PL1))
        gammas_list.append(g)

    return tpp, fpp, err, gammas_list


def generate_data(samples):
    Data0 = np.zeros((2, int(PL0 * samples))).T
    Label0 = np.full((int(PL0 * samples), 1), 0)
    for i in range(int(PL0 * samples)):
        choice = random.random()
        if choice <= W01:
            Data0[i] = (np.random.multivariate_normal(M01, CIJ, 1))
        else:
            Data0[i] = (np.random.multivariate_normal(M02, CIJ, 1))
    Data0 = np.concatenate((Data0, Label0), axis=1)

    Data1 = np.zeros((2, int(PL1 * samples))).T
    Label1 = np.full((int(PL1 * samples), 1), 1)
    for i in range(int(PL1 * samples)):
        choice = random.random()
        if choice <= W11:
            Data1[i] = np.random.multivariate_normal(M11, CIJ, 1)
        else:
            Data1[i] = np.random.multivariate_normal(M12, CIJ, 1)
    Data1 = np.concatenate((Data1, Label1), axis=1)

    Data = np.concatenate((Data0, Data1), axis=0)
    return_data = pd.DataFrame(Data)

    return return_data


def train_classifier_grad_desc(x_train, y_train, x_test, y_test, a, loop_iterations=1000):
    # Code adapted from in class notes, matlab code and practice section on Google Drive
    ones_list = np.ones(x_train.shape[0]).T
    x_train_use = np.asarray(x_train)
    x_test_use = np.asarray(x_test)
    z = np.concatenate(([ones_list], x_train_use.T), axis=0)

    w = np.zeros((3, 1))

    # Need to use .dot product for element wise multiplication of weights and x values

    for loop in range(loop_iterations):
        h = 1 / (1 + np.exp(-1 * np.dot(w.T, z)))
        cost1 = 1 / z.shape[1]
        cost2 = np.dot(z, (h - y_train).T)
        cost = cost1 * cost2
        w = w - a * cost

    z = np.concatenate(([np.ones(x_test.shape[0])], x_test_use.T), axis=0)

    h = 1 / (1 + np.exp(-1 * np.dot(w.T, z)))
    decisions = np.zeros((1, x_test_use.shape[0]))
    decisions = (h >= 0.5)

    tp = [tchoice for tchoice in range(x_test.shape[0]) if (y_test[tchoice] == 0 and decisions[0, tchoice] == 0)]
    fp = [fchoice for fchoice in range(x_test.shape[0]) if (y_test[fchoice] == 1 and decisions[0, fchoice] == 1)]
    print(f'Error on Training Set of Size {x_train.shape[0]}: {100 - ((len(tp) + len(fp)) / 100)} %')
    return w, decisions


def plot_data(eVal, gVal, dVal, D10KVal, RD20, RD200, RD2K):
    min_error = min(eVal)
    error_prob = np.asarray(eVal)
    gamma_location = np.where(error_prob == min_error)[0][0]

    print(f'Gamma Value Selected: {gVal[gamma_location]}')
    print(f'Empirical Gamma: {THEORETICAL_GAMMA}')
    print(f"Minimum Error = {min_error}")
    min_error_round = round(min_error, 3)
    print(f"Minimum Error Index = {gamma_location}")

    gammas_list = list(gVal)
    sel_gamma_results = (dVal >= np.log(gammas_list[gamma_location])).astype(int)

    plt.figure(1)
    plt.title('ROC Curve for ERM Classification On 10K Validation Set')
    plt.plot(false_positive10K, true_positive10K, label="ROC CURVE")
    plt.plot(false_positive10K[gamma_location], true_positive10K[gamma_location], 'go', label=f"Experimental Gamma Minimum Error: {min_error_round}")
    # Calculate for theoretical minimum error
    theo_true_positive = (dVal[int(PL0 * len(list(dVal))):] >= THEORETICAL_GAMMA).sum() / (PL1 * len(list(dVal)))
    theo_false_positive = (dVal[:int(PL0 * len(list(dVal)))] >= THEORETICAL_GAMMA).sum() / (PL0 * len(list(dVal)))
    theo_min_error = (theo_false_positive * PL0) + ((1 - theo_true_positive) * PL1)
    theo_min_error_round = round(theo_min_error, 3)
    plt.plot(theo_false_positive, theo_true_positive, 'D', label=f'Theoretical Gamma Minimum Error: {theo_min_error_round}')
    plt.xlabel('False Detection')
    plt.ylabel('Correct Detection')
    plt.legend()

    correct_0 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 0 and sel_gamma_results[i] == 0)]
    incorrect_0 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 0 and sel_gamma_results[i] == 1)]
    correct_1 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 1 and sel_gamma_results[i] == 1)]
    incorrect_1 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 1 and sel_gamma_results[i] == 0)]

    plt.figure(2)
    plt.title('Correct vs. Incorrect Point Identification, Minimum Probability of Error')
    plt.scatter(D10KVal.values[correct_0, 0], D10KVal.values[correct_0, 1], marker='v', color='g', label=f'Correct Label 0')
    plt.scatter(D10KVal.values[incorrect_0, 0], D10KVal.values[incorrect_0, 1], marker='v', color='r', label=f'Incorrect Label 0')
    plt.scatter(D10KVal.values[correct_1, 0], D10KVal.values[correct_1, 1], marker='+', color='g', label=f'Correct Label 1')
    plt.scatter(D10KVal.values[incorrect_1, 0], D10KVal.values[incorrect_1, 1], marker='+', color='r', label=f'Incorrect Label 1')

    plt.figure(3)

    D20correct_0 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 0 and RD20[0, i] == 0)]
    D20incorrect_0 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 0 and RD20[0, i] == 1)]
    D20correct_1 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 1 and RD20[0, i] == 1)]
    D20incorrect_1 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 1 and RD20[0, i] == 0)]

    plt.title('Correct vs. Incorrect Point Identification, 20 Training Samples')
    plt.scatter(D10KVal.values[D20correct_0, 0], D10KVal.values[D20correct_0, 1], marker='v', color='g', label=f'Correct Label 0')
    plt.scatter(D10KVal.values[D20incorrect_0, 0], D10KVal.values[D20incorrect_0, 1], marker='v', color='r', label=f'Incorrect Label 0')
    plt.scatter(D10KVal.values[D20correct_1, 0], D10KVal.values[D20correct_1, 1], marker='+', color='g', label=f'Correct Label 1')
    plt.scatter(D10KVal.values[D20incorrect_1, 0], D10KVal.values[D20incorrect_1, 1], marker='+', color='r', label=f'Incorrect Label 1')

    plt.figure(4)

    D200correct_0 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 0 and RD200[0, i] == 0)]
    D200incorrect_0 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 0 and RD200[0, i] == 1)]
    D200correct_1 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 1 and RD200[0, i] == 1)]
    D200incorrect_1 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 1 and RD200[0, i] == 0)]

    plt.title('Correct vs. Incorrect Point Identification, 200 Training Samples')
    plt.scatter(D10KVal.values[D200correct_0, 0], D10KVal.values[D200correct_0, 1], marker='v', color='g', label=f'Correct Label 0')
    plt.scatter(D10KVal.values[D200incorrect_0, 0], D10KVal.values[D200incorrect_0, 1], marker='v', color='r', label=f'Incorrect Label 0')
    plt.scatter(D10KVal.values[D200correct_1, 0], D10KVal.values[D200correct_1, 1], marker='+', color='g', label=f'Correct Label 1')
    plt.scatter(D10KVal.values[D200incorrect_1, 0], D10KVal.values[D200incorrect_1, 1], marker='+', color='r', label=f'Incorrect Label 1')

    plt.figure(5)

    D2000correct_0 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 0 and RD2K[0, i] == 0)]
    D2000incorrect_0 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 0 and RD2K[0, i] == 1)]
    D2000correct_1 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 1 and RD2K[0, i] == 1)]
    D2000incorrect_1 = [i for i in range(D10KVal.values.shape[0]) if (D10KVal.values[i, 2] == 1 and RD2K[0, i] == 0)]

    plt.title('Correct vs. Incorrect Point Identification, 2000 Training Samples')
    plt.scatter(D10KVal.values[D2000correct_0, 0], D10KVal.values[D2000correct_0, 1], marker='v', color='g', label=f'Correct Label 0')
    plt.scatter(D10KVal.values[D2000incorrect_0, 0], D10KVal.values[D2000incorrect_0, 1], marker='v', color='r', label=f'Incorrect Label 0')
    plt.scatter(D10KVal.values[D2000correct_1, 0], D10KVal.values[D2000correct_1, 1], marker='+', color='g', label=f'Correct Label 1')
    plt.scatter(D10KVal.values[D2000incorrect_1, 0], D10KVal.values[D2000incorrect_1, 1], marker='+', color='r', label=f'Incorrect Label 1')

    plt.show()


if __name__ == '__main__':
    D20Train = generate_data(20)
    D200Train = generate_data(200)
    D2000Train = generate_data(2000)
    D10KValidate = generate_data(10000)

    # Generate Discriminant scores = p(x|L = 1)/p(x|L = 0)
    disc10K = get_discriminant(D10KValidate)
    true_positive10K, false_positive10K, error10K, gammas_10K = erm_classify(disc10K)

    weightsD20, resultsD20 = train_classifier_grad_desc(D20Train.values[:, :2], D20Train.values[:, 2], D10KValidate.values[:, :2], D10KValidate.values[:, 2], 0.0001, 100)
    weightsD200, resultsD200 = train_classifier_grad_desc(D200Train.values[:, :2], D200Train.values[:, 2], D10KValidate.values[:, :2], D10KValidate.values[:, 2], 0.0001, 100)
    weightsD2000, resultsD2000 = train_classifier_grad_desc(D2000Train.values[:, :2], D2000Train.values[:, 2], D10KValidate.values[:, :2], D10KValidate.values[:, 2], 0.0001, 100)
    print(f'_________________________________________')
    print(f'Part A Outputs and Checks')
    print(f"True Positive Max = {max(true_positive10K)}")
    print(f"True Positive Min = {min(true_positive10K)}")
    print(f"False Positive Max = {max(false_positive10K)}")
    print(f"False Positive Min = {min(false_positive10K)}")

    plot_data(error10K, gammas_10K, disc10K, D10KValidate, resultsD20, resultsD200, resultsD2000)
