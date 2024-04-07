import matplotlib.pyplot as plt
from matrix_sensing import *
import numpy as np

# Independent variables
d = 50                          # Size of ground truth matrix
r = 5                           # Rank of ground truth matrix
c = 5                          # For m = c*d*r
max_epochs = 1000 #10*10**3
learning_rate = 0.001

# initialize variables, generate X, Ais, and labels y
m = c*d*r
print("d, r, m:", d, r, m)
print("Generating training data... ", end="")
U_true, X = make_low_rank_matrix(d, r)
Xnp = X.numpy()
U,D,V = np.linalg.svd(Xnp)
U = U[:, :r]

#Xnp = Xnp / np.linalg.norm(X, 'fro')
Ais = random_RIPs(d, count=m)
ytrue = do_measurement(Ais, X, False)
print("Done.")
print("Generating validation data... ", end="")

val_Ais = random_RIPs(d, count=m)
val_ytrue = do_measurement(val_Ais, X, False)
print("Done.")

# Intialize model related variables and objects
alpha_fn = 0.01
net = MatrixSensing(d, use_factorization=True, error_function='mse', alpha_fn=U_true)
SGDoptimiser = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.000)

# Training related variables
# max_loss_acceptable = 10**-2
# loss_improvement_tolerance = 10**-3
max_epochs_10_perc = 10

print("Training for", max_epochs, "epochs with lr:", learning_rate)
train_loss = 0
bce_loss = 0
for epoch in range(0, max_epochs):
    # Print stats during training
    if (epoch % max_epochs_10_perc == 0):
        val_yhat = net.forward(val_Ais)
        val_loss = mse_loss(val_ytrue, val_yhat)

        model_U = net.U.weight.t().detach().numpy()
        train_U = model_U @ model_U.T
        #train_U = train_U #/ np.linalg.norm(train_U, 'fro')
        
        proj = np.linalg.norm((np.eye(d) - U @ U.T) @ train_U, 'fro')
        diff = np.linalg.norm(Xnp - train_U, 'fro')
        print(f'Epoch: {epoch} Train {train_loss: .10f} Val {val_loss: .10f} Distance {diff: .10f} Proj {proj: .10f}')

    yhat, bce_loss = train(net, Ais, ytrue, SGDoptimiser)
    train_loss = mse_loss(ytrue, yhat)

test_Ais = random_RIPs(X.shape[0], count=m)
test_ytrue = do_measurement(test_Ais, X, True)
test_yhat = net.forward(val_Ais)
test_loss = mse_loss(test_ytrue, test_yhat)
print("Test", test_loss)

