import numpy as np
from proj1_helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w # Initialization of the weights
    loss = 50000000
    for n_iter in range(max_iters):
        loss = compute_loss(y,tx,w) # calculate the MSE loss
        gradient = compute_gradient(y,tx,w) # calculate the gradient
        w = w - gamma * gradient # conduct a step of gradient descent
    return w, loss

# Implementation of the stochastic gradient descent algorithm
# max_iters = the maximum number of repetitions the algorithm is allowed to do
# gamma = the step of the function in the direction of the gradient
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w # intialize the weights for the first iteration
    loss=50000000
    for n_iter in range(max_iters):
        loss = compute_loss(y,tx,w) # compute MSE loss for all samples
        stoch_gradient = compute_stoch_gradient(y,tx,w) # calculate stochastic gradient
        w = w - gamma * stoch_gradient # update the weights using the stochastic gradient
    return w,loss

# least squares minimizer using normal equations.
def least_squares(y, tx):
    # calculate the forcing term and the coefficient matrix respectively
    forcing_term = np.transpose(tx) @ y
    coefficient_matrix = np.transpose(tx) @ tx

    w = np.linalg.solve(coefficient_matrix, forcing_term) # solve the linear equation for w
    loss = compute_loss(y,tx,w)
    return w,loss

# Use the ridge regression formula
def ridge_regression(y, tx, lambda_):
    N = tx.shape #get the dimensions of x
    lambda_prime = 2 * N[0] * lambda_ #calculate the new lambda of the gradient formula
    # calculate the coefficient matrix and the forcing term respectively
    coefficient_matrix = np.transpose(tx) @ tx + lambda_prime * np.eye(N[1])
    forcing_term = np.transpose(tx) @ y

    w = np.linalg.solve(coefficient_matrix, forcing_term) #calculate the w with the ridge normal equation
    loss = compute_loss(y,tx,w)
    return w,loss

# Use the gradient descent method for the
def logistic_regression(y, tx, w_initial, gamma, max_iters):
    w = w_initial #set initial weight for the first iteration
    for iter in range(max_iters):
        grad = compute_gradient(y, tx, w) # calculate the gradient for a single iteration of the algorithm
        w = w - gamma * grad # upate the weights given the gradient and the step
        if iter %25 == 0:
            gamma = gamma/2 # decrease the gamma after some iterations pass to increase the accuracy
        loss = compute_loss(y, tx, w) # calculate loss
    return w, loss

def reg_logistic_regression(y, tx, w_initial, gamma, max_iters, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    threshold = 1e-8
    losses = []
    drop = 0.5
    iter_drop = 25
    w = w_initial
    for iter in range(max_iters):
        grad = compute_gradient(y, tx, w) + 2*lambda_*w
        w = w - gamma * grad
        loss = compute_loss(y, tx, w) + lambda_*np.linalg.norm(w) ** 2
        losses.append(loss)
        if iter % iter_drop == 0:
            gamma = gamma * drop ** np.floor((1+iter) / (iter_drop))
            #print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss
