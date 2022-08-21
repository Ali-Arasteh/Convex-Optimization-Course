import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

X_train = pd.read_csv("X_train.csv").to_numpy()
Y_train = pd.read_csv("Y_train.csv").to_numpy()
la = 0.02
step_size = 0.005
N_epoch = 10000
f_star = 49.9649
groups = [[0], [1], [2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14], [15], [16], [17]]

beta = np.random.normal(0, 1, X_train.shape[1])
beta = np.reshape(beta, (X_train.shape[1], 1))
b = np.random.normal()
f = np.zeros(N_epoch)
for i in range(N_epoch):
    beta_prim = beta-step_size*(1/X_train.shape[0]*np.matmul(X_train.T, np.matmul(X_train, beta)+b-Y_train))
    for group in groups:
        beta[group] = (1-la*np.sqrt(len(group))*step_size/max(np.linalg.norm(beta_prim[group]), la*np.sqrt(len(group))*step_size))*beta_prim[group]
        f[i] += la*np.sqrt(len(group))*np.linalg.norm(beta[group])
    f[i] += 1/(2*X_train.shape[0])*np.linalg.norm(np.matmul(X_train, beta)+b-Y_train)**2
    b = b-step_size*(1/X_train.shape[0]*np.sum(np.matmul(X_train, beta)+b-Y_train))
plt.figure()
plt.semilogy(range(1, N_epoch+1), f-f_star)
plt.title("Group LASSO" + ", " + "Lambda = " + str(la) + ", " + "Step size = " + str(step_size))
plt.savefig("Group LASSO" + ", " + "Lambda = " + str(la) + ", " + "Step size = " + str(step_size) + ".jpg")
for group in groups:
    print(beta[group])

beta = np.random.normal(0, 1, X_train.shape[1])
beta = np.reshape(beta, (X_train.shape[1], 1))
b = np.random.normal()
f_LASSO = np.zeros(N_epoch)
for i in range(N_epoch):
    beta_prim = beta-step_size*(1/X_train.shape[0]*np.matmul(X_train.T, np.matmul(X_train, beta)+b-Y_train))
    beta = (1-la*np.sqrt(X_train.shape[1])*step_size/max(np.linalg.norm(beta_prim), la*np.sqrt(X_train.shape[1])*step_size))*beta_prim
    f_LASSO[i] += la*np.sqrt(X_train.shape[1])*np.linalg.norm(beta)
    f_LASSO[i] += 1/(2*X_train.shape[0])*np.linalg.norm(np.matmul(X_train, beta)+b-Y_train)**2
    b = b-step_size*(1/X_train.shape[0]*np.sum(np.matmul(X_train, beta)+b-Y_train))
plt.figure()
plt.semilogy(range(1, N_epoch+1), f-f_star)
plt.title("LASSO" + ", " + "Lambda = " + str(la) + ", " + "Step size = " + str(step_size))
plt.savefig("LASSO" + ", " + "Lambda = " + str(la) + ", " + "Step size = " + str(step_size) + ".jpg")
print(beta)

plt.figure()
plt.semilogy(range(1, N_epoch+1), f_LASSO-f_star)
plt.semilogy(range(1, N_epoch+1), f_LASSO-f_star)
plt.title("Group LASSO, LASSO" + ", " + "Lambda = " + str(la) + ", " + "Step size = " + str(step_size))
plt.savefig("Group LASSO, LASSO" + ", " + "Lambda = " + str(la) + ", " + "Step size = " + str(step_size) + ".jpg")

beta = np.random.normal(0, 1, X_train.shape[1])
beta = np.reshape(beta, (X_train.shape[1], 1))
beta_prev = copy.deepcopy(beta)
b = np.random.normal()
f_acc = np.zeros(N_epoch)
for i in range(N_epoch):
    beta_prim = beta-step_size*(1/X_train.shape[0]*np.matmul(X_train.T, np.matmul(X_train, beta)+b-Y_train))+(i-2)/(i+1)*(beta-beta_prev)
    beta_prev = copy.deepcopy(beta)
    for group in groups:
        beta[group] = (1-la*np.sqrt(len(group))*step_size/max(np.linalg.norm(beta_prim[group]), la*np.sqrt(len(group))*step_size))*beta_prim[group]
        f_acc[i] += la*np.sqrt(len(group))*np.linalg.norm(beta[group])
    f_acc[i] += 1/(2*X_train.shape[0])*np.linalg.norm(np.matmul(X_train, beta)+b-Y_train)**2
    b = b-step_size*(1/X_train.shape[0]*np.sum(np.matmul(X_train, beta)+b-Y_train))
plt.figure()
plt.semilogy(range(1, N_epoch+1), f_acc-f_star)
plt.title("Acclerated" + ", " + "Lambda = " + str(la) + ", " + "Step size = " + str(step_size))
plt.savefig("Acclerated" + ", " + "Lambda = " + str(la) + ", " + "Step size = " + str(step_size) + ".jpg")
for group in groups:
    print(beta[group])