import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_train = pd.read_csv("X_train.csv").to_numpy()
Y_train = pd.read_csv("Y_train.csv").to_numpy()
la = 1
batch_size_list = [10, 20, 50, 100]
step_size_list = [10**(-2), 10**(-3), 10**(-4), 10**(-5)]
N_epoch = 500
f_star = 57.0410

for batch_size in batch_size_list:
    for t in step_size_list:
        beta = np.random.normal(0, 1, X_train.shape[1])
        beta = np.reshape(beta, (X_train.shape[1], 1))
        b = np.random.normal()
        f = np.zeros(N_epoch)
        for i in range(N_epoch):
            perm = np.random.permutation(X_train.shape[0])
            X_train = X_train[perm,:]
            Y_train = Y_train[perm]
            for j in range(X_train.shape[0]//batch_size):
                x = X_train[j*batch_size:(j+1)*batch_size, :]
                y = Y_train[j*batch_size:(j+1)*batch_size]
                beta = beta-t*(1/batch_size*np.matmul(x.T, np.matmul(x, beta)+b-y)+la*beta)
                b = b-t*(1/batch_size*np.sum(np.matmul(x, beta)+b-y))
            f[i] = 1/(2*X_train.shape[0])*np.linalg.norm(np.matmul(X_train, beta)+b-Y_train)**2+la/2*np.linalg.norm(beta)**2
        plt.figure()
        plt.semilogy(range(1, N_epoch+1), f-f_star)
        plt.title("Batch size = " + str(batch_size) + ", " + "Step size = " + str(t))
        plt.savefig("Batch size = " + str(batch_size) + ", " + "Step size = " + str(t) + ".jpg")