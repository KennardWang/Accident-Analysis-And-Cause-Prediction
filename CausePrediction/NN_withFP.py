
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import OneHotEncoder


def importCSV(dir, columns):
    data = pd.read_csv(dir, header=None, names=columns)
    return data


def plot_image(data):

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel('District')
    ax.set_ylabel('Time')

    ax.scatter(data['District'], data['Time'], s=50, c='b', marker='o', label='Cause1')
    ax.scatter(data['District'], data['Time'], s=50, c='b', marker='x', label='Cause2')
    ax.scatter(data['District'], data['Time'], s=50, c='r', marker='o', label='Cause3')
    ax.scatter(data['District'], data['Time'], s=50, c='r', marker='x', label='Cause4')
    ax.legend()

    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)

    return a1, z2, a2, z3, h


def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.mat(X)
    y = np.mat(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.mat(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.mat(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    return J


def back_propagate(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.mat(X)
    y = np.mat(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.mat(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.mat(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    # perform backpropagation
    for t in range(m):
        a1t = a1[t, :]  # (1, 3)
        z2t = z2[t, :]  # (1, 3)
        a2t = a2[t, :]  # (1, 4)
        ht = h[t, :]  # (1, 4)
        yt = y[t, :]  # (1, 4)

        d3t = ht - yt  # (1, 4)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 4)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 4)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad

if __name__ == '__main__':

    # initialization
    columns = ['District', 'Time', 'Cause']
    data = importCSV('featureDataSet.csv', columns)
    row_num = data.shape[0]
    col_num = data.shape[1]
    flag_num = 4

    x_mat = np.array(data.iloc[:, 0:col_num - 1])
    y_mat = np.array(data.iloc[:, col_num - 1:col_num])

    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y_mat)
    #print(y_onehot.shape)

    # 初始化设置
    input_size = 2
    hidden_size = 3
    num_labels = 4
    learning_rate = 1

    # 随机初始化完整网络参数大小的参数数组
    params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

    m = x_mat.shape[0]
    X = np.mat(x_mat)
    y = np.mat(y_mat)

    # 将参数数组解开为每个层的参数矩阵
    theta1 = np.mat(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.mat(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    #print(theta1.shape, theta2.shape)

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    #print(a1.shape, z2.shape, a2.shape, z3.shape, h.shape)

    c = cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
    #print(c)

    J, grad = back_propagate(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
    #print(J, grad.shape)

    # minimize the objective function
    fmin = opt.minimize(fun=back_propagate, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                    method='TNC', jac=True, options={'maxiter': 250})

    print(fmin)

    X = np.mat(X)
    theta1 = np.mat(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.mat(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    y_pred = np.array(np.argmax(h, axis=1) + 1)
    print(y_pred)

    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('accuracy = {0}%'.format(accuracy * 100))
