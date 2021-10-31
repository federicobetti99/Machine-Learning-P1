import numpy as np
from proj1_helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    This function implements the gradient descent method for linear regression and for MSE as an objective function
    :param y: the labels of the training set
    :param tx: the feature matrix of the training set
    :param initial_w: the initial weight vector, from which we start to iterate
    :param max_iters: the maximum iterations that the method can do
    :param gamma: the learning rate of the gradient method
    """
    w = initial_w # Initialization of the weights
    loss = None
    for n_iter in range(max_iters):
        loss = compute_loss(y,tx,w) # calculate the MSE loss
        gradient = compute_gradient(y,tx,w) # calculate the gradient
        w = w - gamma * gradient # conduct a step of gradient descent
    return w, loss

# Implementation of the stochastic gradient descent algorithm
# max_iters = the maximum number of repetitions the algorithm is allowed to do
# gamma = the step of the function in the direction of the gradient
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    This function implements the stochastic gradient descent method for linear regression and for MSE as an objective function
    :param y: the labels of the training set
    :param tx: the feature matrix of the training set
    :param initial_w: the initial weight vector, from which we start to iterate
    :param max_iters: the maximum iterations that the method can do
    :param gamma: the learning rate of the gradient method
    """
    w = initial_w # intialize the weights for the first iteration
    loss=None
    for n_iter in range(max_iters):
        loss = compute_loss(y,tx,w) # compute MSE loss for all samples
        stoch_gradient = compute_stoch_gradient(y,tx,w) # calculate stochastic gradient
        w = w - gamma * stoch_gradient # update the weights using the stochastic gradient
    return w,loss

# least squares minimizer using normal equations.
def least_squares(y, tx):
    """
    This function implements the normal equations for linear regression and for MSE as an objective function
    :param y: the labels of the training set
    :param tx: the feature matrix of the training set
    """
    # calculate the forcing term and the coefficient matrix respectively
    forcing_term = np.transpose(tx) @ y
    coefficient_matrix = np.transpose(tx) @ tx

    w = np.linalg.solve(coefficient_matrix, forcing_term) # solve the linear equation for w
    loss = compute_loss(y,tx,w)
    return w,loss

# Use the ridge regression formula
def ridge_regression(y, tx, lambda_):
    """
    This function implements the normal equations for ridge_regression and for MSE as an objective function
    :param y: the labels of the training set
    :param tx: the feature matrix of the training set
    :param lambda_: the regularization parameter
    """
    N = tx.shape #get the dimensions of x
    lambda_prime = 2 * N[0] * lambda_ #calculate the new lambda of the gradient formula
    # calculate the coefficient matrix and the forcing term respectively
    coefficient_matrix = np.transpose(tx) @ tx + lambda_prime * np.eye(N[1])
    forcing_term = np.transpose(tx) @ y

    w = np.linalg.solve(coefficient_matrix, forcing_term) #calculate the w with the ridge normal equation
    loss = compute_loss(y,tx,w)
    return w,loss

# Use the gradient descent method for the
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    This function implements the gradient descent method for logistic regression
    :param y: the labels of the training set
    :param tx: the feature matrix of the training set
    :param initial_w: the initial weight vector, from which we start to iterate
    :param max_iters: the maximum iterations that the method can do
    :param gamma: the learning rate of the gradient method
    """
    w = initial_w #set initial weight for the first iteration
    for iter in range(max_iters):
        grad = calculate_gradient(y, tx, w) # calculate the gradient for a single iteration of the algorithm
        w = w - gamma * grad # upate the weights given the gradient and the step
        if iter %25 == 0:
            gamma = gamma/2 # decrease the gamma after some iterations pass to increase the accuracy
        loss = calculate_loss(y, tx, w) # calculate loss
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    This function implements the gradient descent method for regularized logistic regression
    :param y: the labels of the training set
    :param tx: the feature matrix of the training set
    :param lambda_: the regularization parameter for the function
    :param initial_w: the initial weight vector, from which we start to iterate
    :param max_iters: the maximum iterations that the method can do
    :param gamma: the learning rate of the gradient method
    """
    threshold = 1e-8
    losses = []
    drop = 0.5
    iter_drop = 25
    w = initial_w
    for iter in range(max_iters):
        grad = calculate_gradient(y, tx, w) + 2*lambda_*w
        w = w - gamma * grad
        loss = calculate_loss(y, tx, w) + lambda_*np.linalg.norm(w) ** 2
        losses.append(loss)
        if iter % iter_drop == 0:
            gamma = gamma * drop ** np.floor((1+iter) / (iter_drop))
            #print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss
