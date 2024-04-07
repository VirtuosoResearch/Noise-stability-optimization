'''
Matrix Sensing using PyTorch
'''
import numpy as np
import numpy.linalg as npla

def make_low_rank_matrix(d, r):
    U = np.random.normal(0, 1, (d, r))
    return U, U @ U.T

def random_RIPs(d, count):
    rip = np.random.normal(0, 1, size=(count, d, d))
    inds = range(count)
    rip[inds, :, :] = 0.5*(rip[inds, :, :] + np.transpose(rip, (0, 2, 1))[inds, :, :])
    return rip

def do_measurement(Ais, X, d):
    reshaped_A = Ais.reshape(-1, d * d)
    reshaped_X = X.reshape(d * d, 1)
    return reshaped_A @ reshaped_X
    #Ais.
    #traces = torch.zeros(Ais.shape[0])
    #if ((len(Ais.shape) != len(X.shape)+1)
    #    or len(X.shape) != 2):
    #    print("Arguments should be of the form (batch_size, n, n) and (n, n).")
    #    print("Received: {} and {}".format(Ais.shape, X.shape))
    #    return traces

    #m, d, _ = Ais.shape
    #Ais_reshape, X_reshape = Ais.view((m, d*d)), X.reshape(d*d, 1)
    #traces = torch.mm(Ais_reshape.float(), X_reshape.float()).view((m,))
    return traces

def get_prob_label(Ais, X):
    y = batch_matrix_ip(Ais, X)
    y = 1.0 / (1 + torch.exp(-y))
    probs = torch.rand(y.shape).type(torch.DoubleTensor)
    y_label = (probs < y).type(torch.DoubleTensor)
    return y_label

def train_GD(Ais, y_true, U, lr, d):
    grad = get_grad(Ais, y_true, U, d)
    U = U - lr * grad
    return U

def train_NSO(Ais, y_true, U, lr, perturb, d, use_negative = True):
    V = np.random.normal(0, 1, (d, d)) * perturb

    plus_grad  = get_grad(Ais, y_true, U + V, d)
    if use_negative:
        minus_grad = get_grad(Ais, y_true, U - V, d)
    else:
        minus_grad = plus_grad

    U = U - lr * (plus_grad + minus_grad) / 2
    return U


def get_grad(Ais, y_true, U, d):
    y_pred = do_measurement(Ais, U @ U.T, d)

    reshaped_A = Ais.reshape(-1, d * d)
    convolved_A = np.diag((y_pred - y_true).flatten()) @ reshaped_A
    convolved_A = np.sum(convolved_A, axis = 0) / len(y_true)
    convolved_A = convolved_A.reshape(d, d)

    grad = convolved_A @ U
    return grad

def mse_loss(ytrue, yhat):
    return np.mean((yhat - ytrue) ** 2) / np.mean(ytrue ** 2)
