import matplotlib.pyplot as plt
from matrix_sensing import *
import numpy as np

# Independent variables
d = 100                          # Size of ground truth matrix
r = 5                           # Rank of ground truth matrix
c = 5                          # For m = c*d*r
use_factorization = True        # Does the model use UU^T or just U
learning_rate = 0.0001

# initialize variables, generate X, Ais, and labels y
# initialize variables, generate X, Ais, and labels y
m = c*d*r
U_true, X = make_low_rank_matrix(d, r)
Ais = random_RIPs(X.shape[0], count=m)
y_true = do_measurement(Ais, X, d)

val_Ais = random_RIPs(X.shape[0], count=m)
val_ytrue = do_measurement(val_Ais, X, d)

# Intialize model related variables and objects
init_scale = 1.00
lr = 0.0025
max_epochs_10_perc = 20
U = np.random.randn(d, d) * init_scale

perturb = 0.02
max_epochs = 20001
perturb_flag = False
print('max epochs', max_epochs, lr, 'perturb', perturb, perturb_flag)
for epoch in range(1, max_epochs):
    U = train_NSO(Ais, y_true, U, lr, perturb, d, perturb_flag)
    y_pred = do_measurement(Ais, U @ U.T, d)
    train_loss = mse_loss(y_true, y_pred)

    # Print stats during training
    if (epoch % max_epochs_10_perc == 0):
        val_yhat = do_measurement(val_Ais, U @ U.T, d)
        val_loss = mse_loss(val_ytrue, val_yhat)

        diff = np.linalg.norm(X - U @ U.T, 'fro') / np.linalg.norm(X, 'fro')
        print("Epoch", epoch, "Training", train_loss, "Validation", val_loss, "Distance", diff)
