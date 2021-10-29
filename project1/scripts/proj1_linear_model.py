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
    losses, ws = least_squares_GD(y_training, x_training_augmented, np.zeros(x_training_augmented.shape[1]) , 2000, gamma)
    w_opt_training = ws[-1]
    predictions_test = x_testing_augmented@w_opt_training
    predictions_test = np.array([0 if el <0.5 else 1 for el in predictions_test])
    acc_test = compute_accuracy(y_testing, predictions_test)
    return acc_test

def finetune_GD(tX, y, k_fold = 4, degrees = np.arange(1,5)):
    seed = 1
    testing_acc = np.zeros(len(degrees))
    k_indices = build_k_indices(y, k_fold, seed)
    for index in range(len(degrees)):
        current_sum_test = 0
        for k in range(k_fold):
            current_test_acc = cross_validation_GD(y, tX, k_indices, k, degrees[index], gamma = 5*10e-4)
            current_sum_test += current_test_acc
        testing_acc[index] = current_sum_test / k_fold
    best_result = np.where(testing_acc == np.amax(testing_acc))
    print(testing_acc)
    degree_opt = degrees[best_result[0]]
    print(degree_opt)
    return degree_opt

def optimal_weights_GD(tX,y,degree):
    tX_augmented = build_poly(tX, degree)
    losses, ws = least_squares_GD(y, tX_augmented, np.zeros(tX_augmented.shape[1]) , 2000, 5*10e-4)
    w_opt_GD = ws[-1]
    return w_opt_GD

def predict_GD(tX,w,degree=2):
    # make the predictions with the augmented test set
    #since we trained the model in augmented data, we augment the test set
    tX_augmented = build_poly(tX, degree)
    # make the predictions with the augmented test set and GD
    predictions_GD = tX_augmented @ w
    return predictions_GD
###########################################################
#       Stochastic Gradient Descent                       #
###########################################################

# compute the stochastic gradient of a random sample
def compute_stoch_gradient(y, tx, w):
    N = y.shape[0]# number of samples
    random_number = np.random.randint(0,N)# generate random index
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

def cross_validation_SGD(y, x, k_indices, k, degree, gamma):
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
    losses, ws = least_squares_SGD(y_training, x_training_augmented, np.zeros(x_training_augmented.shape[1]) , 1000, gamma)
    w_opt_training = ws[-1]
    predictions_test = x_testing_augmented@w_opt_training
    predictions_test = np.array([0 if el <0.5 else 1 for el in predictions_test])
    acc_test = compute_accuracy(y_testing, predictions_test)
    return acc_test

def finetune_SGD(tX, y, k_fold = 4, degrees = np.arange(1,5)):
    seed = 1
    testing_acc = np.zeros(len(degrees))
    k_indices = build_k_indices(y, k_fold, seed)
    for index in range(len(degrees)):
        current_sum_test = 0
        for k in range(k_fold):
            current_test_acc = cross_validation_SGD(y, tX, k_indices, k, degrees[index], gamma = 5*10e-4)
            current_sum_test += current_test_acc
        testing_acc[index] = current_sum_test / k_fold
    best_result = np.where(testing_acc == np.amax(testing_acc))
    print(testing_acc)
    degree_opt = degrees[best_result[0]]
    print(degree_opt)
    return degree_opt

def optimal_weights_SGD(tX, y, degree):
    tX_augmented = build_poly(tX, degree)
    losses, ws = least_squares_SGD(y, tX_augmented, np.zeros(tX_augmented.shape[1]) , 2000, 5*10e-4)
    w_opt_SGD = ws[-1]
    return w_opt_SGD

def predict_SGD(tX, w, degree=2):
    # make the predictions with the augmented test set
    #since we trained the model in augmented data, we augment the test set
    tX_augmented = build_poly(tX, degree)
    # make the predictions with the augmented test set and GD
    predictions_SGD = tX_augmented @ w
    return predictions_SGD

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
