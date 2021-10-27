import numpy as np
from proj1_helpers import *

##############################################
#       Loss + Gradient Calculators          #
##############################################

# compute the MSE Loss
def compute_loss(y, tx, w):
    N = y.shape[0] # N = Number of samples
    e = y - tx @ w # e = error vector (truth - prediction)
    loss = 1/(2*N) * np.dot(e,e) # calculate the average loss
    return loss

# compute the gradient for the MSE loss function
def compute_gradient(y, tx, w):
    N = y.shape[0] # N = Number of samples
    e = y - tx @ w # e = error vector (truth - prediction)
    gradient = -(1/N) * (tx.T) @ (e) # calculate the gradient
    return gradient


##############################################
#          Gradient Descent                  #
##############################################

# do the gradient descent algorithm
# max_iters = the maximum number of repetitions the algorithm is allowed to do
# gamma = the step of the function in the direction of the gradient
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]# A list of all the weights
    losses = []
    w = initial_w # Initialization of the weights
    for n_iter in range(max_iters):
        loss = compute_loss(y,tx,w) # calculate the MSE loss
        gradient = compute_gradient(y,tx,w) # calculate the gradient
        w = w - gamma * gradient # conduct a step of gradient descent
        ws.append(w) # append the current weight
        losses.append(loss) # compute the next loss
        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        # bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

# Use cross validation to determine optimal data augmenttion
# cross validation function for gradient descent
# k-indices = random subsets of the original samples
# k the set that we will use as the test set out of the subsets
# degree = the degree up to which we will exponentiate each feature
def cross_validation_GD(y, x, k_indices, k, degree, gamma):
    """return the loss of ridge regression."""
    N = y.shape[0]
    k_fold = k_indices.shape[0]
    list_ = []
    interval = int(N/k_fold)
    for i in range(k_fold):
        if i != k:
            list_.append(i)
    x_training = np.zeros((int((k_fold-1)/k_fold*N), x.shape[1]))
    y_training = np.zeros(int((k_fold-1)/k_fold*N))
    for j in range(len(list_)):
        x_training[interval*(j):interval*(j+1), :] = x[np.array([k_indices[list_[j]]]), :]
    x_testing = x[k_indices[k], :]
    for j in range(len(list_)):
        y_training[interval*(j):interval*(j+1)] = y[np.array([k_indices[list_[j]]])]
    y_testing = y[k_indices[k]]
    x_training_augmented = build_poly(x_training, degree)
    x_testing_augmented = build_poly(x_testing, degree)
    losses, ws = least_squares_GD(y_training, x_training_augmented, np.zeros(x_training_augmented.shape[1]) , 1000, gamma)
    w_opt_training = ws[-1]
    print(losses)
    #loss_tr = compute_loss(y_training, x_training_augmented, w_opt_training)
    #loss_te = compute_loss(y_testing, x_testing_augmented, w_opt_training)
    predictions_test = x_testing_augmented@w_opt_training
    print(predictions_test)
    predictions_test = np.array([0 if el <0.5 else 1 for el in predictions_test])
    print(predictions_test)
    print(y_testing)
    acc_test = compute_accuracy(y_testing, predictions_test)
    return acc_test

###########################################################
#       Stochastic Gradient Descent                       #
###########################################################

# compute the stochastic gradient of a random sample
def compute_stoch_gradient(y, tx, w):
    N = y.shape[0]# number of samples
    random_number = random.randint(0,N)# generate random index
    xn = tx[random_number,:]# get sample of that index
    random_gradient = - np.dot(xn, y[random_number] - np.dot(xn,w))# calculate the stochastic gradient
    return random_gradient

# Implementation of the stochastic gradient descent algorithm
# max_iters = the maximum number of repetitions the algorithm is allowed to do
# gamma = the step of the function in the direction of the gradient
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]                                                      #initialize weight list
    losses = [] # initialize list of losses
    w = initial_w # intialize the weights for the first iteration
    for n_iter in range(max_iters):
        loss = compute_loss(y,tx,w) # compute MSE loss for all samples
        stoch_gradient = compute_stoch_gradient(y,tx,w) # calculate stochastic gradient
        w = w - gamma * stoch_gradient # update the weights using the stochastic gradient
        ws.append(w) # append the next weight
        losses.append(loss) # append the current loss to the list
        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

############################################################
#         Normal Equations                                 #
############################################################

# least squares minimizer using normal equations.
def least_squares(y, tx):
    # calculate the forcing term and the coefficient matrix respectively
    forcing_term = np.transpose(tx) @ y
    coefficient_matrix = np.transpose(tx) @ tx

    w = np.linalg.solve(coefficient_matrix, forcing_term) # solve the linear equation for w
    return w

# do we need this?
def test_your_least_squares(y, tx):
    """compare the solution of the normal equations with the weights returned by gradient descent algorithm."""
    w_least_squares = least_squares(y, tx)
    initial_w = np.zeros(tx.shape[1])
    max_iters = 50
    gamma = 0.7
    losses_gradient_descent, w_gradient_descent = gradient_descent(y, tx, initial_w, max_iters, gamma)
    w = w_gradient_descent[-1]
    err = np.linalg.norm(w_least_squares-w)
    return err
