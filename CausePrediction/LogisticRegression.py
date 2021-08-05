
# multiple classification with Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


def importCSV(dir, columns):
    data = pd.read_csv(dir, header=None, names=columns)
    data.insert(0, 'Ones', 1)
    return data


def plot_image(data):

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(data['District'], data['Time'], s=50, c='b', marker='o', label='Cause1')
    ax.scatter(data['District'], data['Time'], s=50, c='b', marker='x', label='Cause2')
    ax.scatter(data['District'], data['Time'], s=50, c='r', marker='o', label='Cause3')
    ax.scatter(data['District'], data['Time'], s=50, c='r', marker='x', label='Cause4')
    ax.legend()
    ax.set_xlabel('District')
    ax.set_ylabel('Time')
    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, l):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)

    first_term = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second_term = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg_term = (l / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first_term - second_term) / len(X) + reg_term


def gradient(theta, X, y, l):

    # matrix
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)

    times = int(theta.ravel().shape[1])
    grad = np.zeros(times)
    error = sigmoid(X * theta.T) - y

    for i in range(times):
        tmp = np.multiply(error, X[:,i])

        if i==0:
            grad[i] = np.sum(tmp)/len(X)
        else:
            grad[i] = np.sum(tmp)/len(X) + l/len(X) * theta[:,1]

    return grad


def logistic_regression(X, y, num_labels, l):
    times = X.shape[1]
    all_theta = np.zeros((num_labels, times))

    for i in range(1, num_labels+1):
        theta = np.zeros(times)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = y_i.reshape(X.shape[0],1)

        # minimize
        fmin = opt.minimize(fun=cost, x0=theta, args=(X,y_i,l), method='TNC', jac=gradient)
        all_theta[i-1,:] = fmin.x

    return all_theta


def predict(X, all_theta):
    X = np.mat(X)
    all_theta = np.mat(all_theta)
    h = sigmoid(X * all_theta.T)

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    #print(h_argmax)
    return h_argmax


if __name__ == '__main__':

    # initialization
    columns = ['District', 'Time', 'Cause']
    data = importCSV('featureDataSet.csv', columns)
    row_num = data.shape[0]
    col_num = data.shape[1]
    flag_num = 4
    x_mat = np.array(data.iloc[:, 0:col_num-1])
    y_mat = np.array(data.iloc[:, col_num-1:col_num])

    # draw
    plot_image(data)

    # analysis
    all_theta = logistic_regression(x_mat,y_mat,flag_num,1)
    #print(all_theta)

    y_pred = predict(x_mat,all_theta)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y_mat)]
    accuracy = sum(map(int, correct)) / float(len(correct))

    #print(correct)
    print('accuracy = {0}%'.format(accuracy * 100))
