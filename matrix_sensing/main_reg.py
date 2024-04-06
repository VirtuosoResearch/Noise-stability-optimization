import matplotlib.pyplot as plt
from matrix_sensing import *
import numpy as np

# Independent variables
d = 100                          # Size of ground truth matrix
r = 5                           # Rank of ground truth matrix
c = 5                          # For m = c*d*r
is_classification = False#True        # Use this to control regression/classification
use_factorization = True        # Does the model use UU^T or just U
max_epochs = 10000 #10*10**3
learning_rate = 0.001

# initialize variables, generate X, Ais, and labels y
m = c*d*r
print("d, r, m:", d, r, m)
print("Generating training data... ", end="")
#X, U, D, V = make_low_rank_matrix(d, r)
X = make_low_rank_matrix(d, r)
Xnp = X.numpy()
Ais = random_RIPs(X.shape[0], symmetric=True, count=m)
ytrue = do_measurement(Ais, X, False)
ytrue_squared_average = torch.sum(ytrue**2)
print("Done.")
print("Generating validation data... ", end="")

val_Ais = random_RIPs(X.shape[0], symmetric=True, count=m)
val_ytrue = do_measurement(val_Ais, X, False)
val_ytrue_squared_average = torch.sum(val_ytrue**2)
print("Done.")

# Intialize model related variables and objects
activate = is_classification
error_fn = 'mse'
alpha_fn = 1.00
net = MatrixSensing(d, use_factorization=True, activate=False, error_function='mse', alpha_fn=alpha_fn)
SGDoptimiser = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.0)

# Training related variables
# max_loss_acceptable = 10**-2
# loss_improvement_tolerance = 10**-3
max_epochs_10_perc = 20

print("Training for", max_epochs, "epochs with lr:", learning_rate)
for epoch in range(1, max_epochs):
    yhat = train(net, Ais, ytrue, SGDoptimiser)
    train_loss = mse_loss(ytrue, yhat)

    # Print stats during training
    if (epoch % max_epochs_10_perc == 0):
        val_yhat = net.forward(val_Ais)
        val_loss = mse_loss(val_yhat, val_ytrue)

        model_U = net.U.weight.t().detach().numpy()
        actual_model_U = model_U @ model_U.T
        diff = np.linalg.norm(Xnp - actual_model_U, 'fro') / np.linalg.norm(Xnp, 'fro')
        print("Epoch", epoch, "Training", train_loss, "Validation", val_loss, "Distance", diff)
