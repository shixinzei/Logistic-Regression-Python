# encoding:utf-8

import numpy as np
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt

'''
Date:2020.7.7
@author: wydxry
'''

def myloaddata(datapath):
    '''

    :param datapath:
    :return:
    '''
    # 导入.arff格式的原始数据
    source = arff.loadarff(datapath)
    df = pd.DataFrame(source[0])
    data = df.values

    # 打乱数据集操作,每次运行时数据排列顺序不一样，能够后面划分训练集测试集时每次运行时都不一样
    # index = [i for i in range(data.shape[0])]
    # np.random.shuffle(index)
    # data = data[index]

    x = data[:, :-1]
    y = data[:, -1]
    Y = np.zeros((y.shape[0], 1))
    for i in range(y.shape[0]):
        Y[i] = (str(y[i]).split('_')[1][0] == 'p')

    # 数据归一化
    mu = 1 / x.shape[0] * np.sum(x, axis=0)
    x = x - mu
    thetafang = 1 / x.shape[0] * np.sum(x ** 2, axis=0)
    X = x / thetafang

    X = np.array(X)
    Y = np.array(Y)
    X = X.T
    Y = Y.T

    # 划分训练集测试集
    X_train = X[:, 1:600]
    Y_train = Y[:, 1:600]
    X_test = X[:, 600:-1]
    Y_test = Y[:, 600:-1]
    return X_train, Y_train, X_test, Y_test


def mysigmoid(x):
    '''
    activation function:
    sigmoid function
    g(z)=1/(1+e**(-z))
    :param x:自变量
    :return:激活函数值
    '''
    return 1.0 / (1.0 + np.e ** (-x))


def myloss(y, y_pre):
    '''
    Loss function:
    L(y,y_pre)=-[(y*log(y_pre)+(1-y)*log(1-y_pre)]
    :param y:真实值
    :param y_pre:预测值
    :return:真实值与预测值直接的误差
    '''
    loss = 0
    for i in range(y.shape[1]):
        if y_pre[0][i] > 0 and (1 - y_pre[0][i]) > 0:
            loss -= y[0][i] * np.log(y_pre[0][i]) + (1 - y[0][i]) * np.log(1 - y_pre[0][i])
    loss /= y.shape[1]
    return loss


def mygradientdescent(x, y, epoch, lr):
    '''

    :param x:
    :param y:
    :param epoch:
    :param lr:
    :return:
    '''
    w = np.random.randn(x.shape[0], 1)
    b = np.zeros((1, 1))
    acc_train_total = []
    acc_test_total = []
    loss_total = []
    i_total = []
    for i in range(1, epoch + 1):
        z = np.dot(w.T, x) + b
        A = mysigmoid(z)
        dz = A - y
        dw = np.dot(x, dz.T)
        db = 1 / x.shape[1] * np.sum(dz)
        w = w - lr * dw
        b = b - lr * db
        if i % 50 == 0:
            loss = myloss(y, A)
            acc_train = np.sum(mypredict(x, w, b) == y[0]) / len(y[0])
            acc_test = np.sum(mypredict(x_test, w, b) == y_test[0]) / len(y_test[0])
            print("iteration: " + str(i) + " loss = " + str(loss) + " Train acc = " + str(acc_train)
                  + " Test acc = " + str(acc_test))
            i_total.append(int(i))
            loss_total.append(loss)
            acc_train_total.append(acc_train)
            acc_test_total.append(acc_test)
    return w, b, loss_total, acc_train_total, acc_test_total, i_total


def mypredict(pre_x, w, b):
    '''

    :param pre_x:
    :param w:
    :param b:
    :return:
    '''
    z = np.dot(w.T, pre_x) + b
    A = mysigmoid(z)
    pre_y = []
    for i in range(A.shape[1]):
        if A[0][i] >= 0.5:
            pre_y.append(1)
        else:
            pre_y.append(0)
    pre_y = np.array(pre_y)
    return pre_y


def myplotresult(w, b, loss_total, acc_train_total, acc_test_total, i_total):
    '''

    :param w:
    :param b:
    :param loss_total:
    :param acc_train_total:
    :param acc_test_total:
    :param i_total:
    :return:
    '''
    # 打印结果:损失值、准确率以及训练结束时的参数
    print("Final weight:")
    for i in range(x_train.shape[0]):
        print("w" + str(i + 1) + " = " + str(w[i][0]))
    print("b" + " = " + str(b[0][0]))
    print("Final loss:" + str(loss_total[-1]))
    # 训练集测试集准确率比较
    print("Compare:")
    print("Train acc = " + str(acc_train_total[-1]))
    print("Test acc = " + str(acc_test_total[-1]))

    # 绘图
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(i_total, loss_total, 'g', lw=3)
    ax1.set_title("Loss value")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(i_total, acc_train_total, 'r', lw=2)
    ax2.plot(i_total, acc_test_total, 'b', lw=2)
    ax2.set_title("Train accuracy and Test accuracy")

    plt.show()


if __name__ == '__main__':
    # 载入数据以及划分数据集
    x_train, y_train, x_test, y_test = myloaddata("diabetes.arff")
    # 设置学习率
    lr = 0.0014
    # 设置迭代次数，即算法训练的周期
    epoch = 8000
    f_w, f_b, loss_total, acc_train_total, acc_test_total, i_total = mygradientdescent(x_train, y_train, epoch, lr)
    # 打印结果:损失值、准确率以及训练结束时的参数
    myplotresult(f_w, f_b, loss_total, acc_train_total, acc_test_total, i_total)


