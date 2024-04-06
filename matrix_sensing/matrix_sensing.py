'''
Matrix Sensing using PyTorch
'''
import numpy as np
import numpy.linalg as npla
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.stats import ortho_group

def sharper_sigmoid(x, sharpness=1):
    return 1/(1+torch.exp(-sharpness*x))

def make_low_rank_matrix(d, r):
    U = np.random.normal(0, 1, (d, r))
    return torch.from_numpy(U).type(torch.DoubleTensor), torch.from_numpy(U @ U.T).type(torch.DoubleTensor)

def random_RIPs(d, count):
    rip = np.random.normal(0, 1, size=(count, d, d))
    inds = range(count)
    rip[inds, :, :] = 0.5*(rip[inds, :, :] + np.transpose(rip, (0, 2, 1))[inds, :, :])
    return torch.from_numpy(rip).type(torch.DoubleTensor)

def batch_matrix_ip(Ais, X):
    traces = torch.zeros(Ais.shape[0])
    if ((len(Ais.shape) != len(X.shape)+1)
        or len(X.shape) != 2):
        print("Arguments should be of the form (batch_size, n, n) and (n, n).")
        print("Received: {} and {}".format(Ais.shape, X.shape))
        return traces

    m, d, _ = Ais.shape
    Ais_reshape, X_reshape = Ais.view((m, d*d)), X.reshape(d*d, 1)
    traces = torch.mm(Ais_reshape, X_reshape).view((m,))
    return traces

def do_measurement(Ais, X, do_sigmoid):
    y = batch_matrix_ip(Ais, X)
    if do_sigmoid:
        y = sharper_sigmoid(y)
        y = y > 0.5
    return y.type(torch.DoubleTensor)

def get_prob_label(Ais, X):
    y = batch_matrix_ip(Ais, X)
    y = 1.0 / (1 + torch.exp(-y))
    probs = torch.rand(y.shape).type(torch.DoubleTensor)
    y_label = (probs < y).type(torch.DoubleTensor)
    return y_label

class MatrixSensing(nn.Module):
    '''
    The Module to perform Matrix Sensing.

    Uses nn.Linear for U and accesses U.weight to introduce factorization
    '''
    def __init__(self, d, use_factorization=True, error_function='bce', alpha_fn=1.0): #1/d**2
        super().__init__()

        self.d = d              # matrix dimension
        self.use_factorization = use_factorization  # use UU^T form?
        self.alpha = 1.0 #alpha_fn

        print("#### Model settings ####")
        print("Factorization :", self.use_factorization)
        print("Error         :", error_function)
        #print("alpha         :", self.alpha)
        print("########")

        # proper init
        self.U = nn.Linear(d, 5, bias=False)
        self.U.weight.data = torch.t(alpha_fn).data.type(torch.DoubleTensor)
        #self.U.weight.data = (0.01 * self.U.weight.data).type(torch.DoubleTensor)

        # full rank init
        #self.U = nn.Linear(d, d, bias=False)
        #nn.init.eye_(self.U.weight)
        #self.U.weight.data = self.U.weight.data.double()
        #self.U.weight.data = 0.005 * self.U.weight.data
        #print(3, self.U.weight.shape, self.U.weight.type())

        # which loss function to use.
        if error_function=='logistic':
            self.error_function = torch.nn.SoftMarginLoss()
        elif error_function=='mse':
            self.error_function = torch.nn.MSELoss()
        elif error_function == 'bce':
            self.error_function = torch.nn.BCEWithLogitsLoss()
        else:
            print("Unrec loss fn")

        # keep track of loss value using this
        self.loss_value = torch.from_numpy(np.array([np.inf])).type(torch.DoubleTensor)
        self.loss_value.requires_grad_(True)

    def forward(self, Ai):
        if self.use_factorization:
            # Note that internally torch stores U already transposed.
            # So we use U.weight.t() when we would want to use the matrix U
            # and we would use U.weight when we want U^T.
            yhat = do_measurement(Ai, torch.mm(self.U.weight.t(), self.U.weight), False)
        else:
            yhat = do_measurement(Ai, self.U.weight.t(), False)

        return yhat

def train(net, Ais, y_input, optimiser, project=False):
    yhat = net.forward(Ais)

    net.loss_value = net.error_function(yhat, y_input)
    optimiser.zero_grad()
    net.loss_value.backward()
    optimiser.step()

    # project
    #if project:
    #    net.U.weight.data = net.U.weight.data / torch.norm(net.U.weight.data)
    return [yhat, net.loss_value]

def train_noise(net, Ais, y_input, optimiser, noise_rate = 0.01):
    #noise_Ai = Ais + noise_rate * torch.randn(Ais.shape).double()
    yhat = net.forward(Ais)
    ytrue = Variable(y_input, requires_grad=False)
    #y_noise = torch.rand(y_input.shape) < noise_rate
    #y_flip = (y_input + y_noise.double()) % 2
    #y_flip = y_flip.double()

    #net.loss_value = net.error_function(yhat, y_flip)
    net.loss_value = net.error_function(yhat, ytrue)
    optimiser.zero_grad()
    net.loss_value.backward()
    optimiser.step()
    return yhat

def binary_loss(ytrue, yhat):
    y_binary = sharper_sigmoid(yhat)
    y_binary = (y_binary > 0.5).type(torch.DoubleTensor)
    return torch.mean(torch.abs(y_binary - ytrue)).item()

def mse_loss(ytrue, yhat):
    return torch.mean((yhat-ytrue)**2).item()/torch.mean(ytrue**2).item()

