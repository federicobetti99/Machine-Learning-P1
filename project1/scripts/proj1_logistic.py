import numpy as np
from proj1_helpers import *

###############################################
#        sigmoid                              #
###############################################

#commpute the sigmoid function for a vector t
def sigmoid(t):
    # seperate the indices where t is negative from the ones where t is non-negative
    positive_indices = np.where(t >= 0)[0]
    negative_indices = np.where(t < 0)[0]

    #calculate the output seperately for the positive values of t and for the negative ones
    # this is done in order to avoid numerical problems s.a. overflow,division by 0 e.t.c.
    z = np.zeros(len(t))
    z[positive_indices] = 1 / (1+np.exp(-t[positive_indices]))
    z[negative_indices] = np.exp(t[negative_indices]) / (1 + np.exp(t[negative_indices]))
    return z

##############################################
#        calculate loss                      #
#        calculate gradient                  #
##############################################

#calculate the loss for the losistic
def calculate_loss(y, tx, w):
    #seperate the indices where the prediction is positive and where the prediction is negative
    pos_ind = np.where(tx @ w >=0)[0]
    neg_ind = np.where(tx @ w <0)[0]
    #calculate the loss for both the positive and the negative indices.
    loss_pos = - y[pos_ind] * (tx @ w)[pos_ind] + (tx @ w)[pos_ind] + np.log(1+np.exp(-(tx @ w)[pos_ind]))# I am a little curious about th minus here
    loss_neg = - y[neg_ind] * (tx @ w)[neg_ind] - (tx @ w)[neg_ind] + np.log(1+np.exp((tx @ w)[neg_ind]))
    return loss_pos.sum() + loss_neg.sum()

# calculate the gradient for logistic regression
def calculate_gradient(y, tx, w):
    return np.transpose(tx) @ (sigmoid(tx @ w) - y)

###############################################
#   gradient descent for logistic regression  #
###############################################

# Use the gradient descent method for the
def learning_by_gradient_descent(y, tx, w_initial, gamma, max_iters):
    losses = [] #list of losses
    w = w_initial #set initial weight for the first iteration
    for iter in range(max_iters):
        grad = calculate_gradient(y, tx, w) # calculate the gradient for a single iteration of the algorithm
        w = w - gamma * grad # upate the weights given the gradient and the step
        if iter %25 == 0:
            gamma = gamma/2 # decrease the gamma after some iterations pass to increase the accuracy
        loss = calculate_loss(y, tx, w) # calculate loss
        losses.append(loss) # append loss to the list of losses
    return losses, w

#############################################
#        newton method                      #
#############################################

# I believe that the second order is used nowhere and should be removed since it is also arithmetically unstable
def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    diag = sigmoid(tx @ w) * (1 - sigmoid(tx @ w))
    D = diag * np.eye(tx.shape[0])
    return np.transpose(tx) @ D @ tx

#calculate the values of the logistic regression
def logistic_regression(y, tx, w):
    grad = calculate_gradient(y, tx, w) #calculate the gradient
    hess = calculate_hessian(y, tx, w) #
    loss = calculate_loss(y, tx, w) # calcualte the loss
    return loss, grad, hess

# Should this be removed as well since we only have to use th gradient method
def learning_by_newton_method(y, tx, w, gamma):
    loss, grad, hess = logistic_regression(y, tx, w)
    sol = np.linalg.solve(hess, grad)
    w = w - gamma * sol
    return loss, w

############################################
#  penalized logistic regression           #
############################################

# Compute the logistic regression with an L2 regularizer
def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    loss = calculate_loss(y, tx, w) + lambda_*np.linalg.norm(w) ** 2
    grad = calculate_gradient(y, tx, w) + 2*lambda_*w
    hess = calculate_hessian(y, tx, w) + 2*lambda_*np.eye(w.shape[0])
    return loss, grad, hess

# Execute one step of regularized logistic regression
def learning_by_penalized_gradient(y, tx, w_initial, gamma, max_iters, lambda_):
    threshold = 1e-8 # threshold to stop execution
    losses = []
    w = w_initial # initialize weights for the first iteration of the algorithm
    for iter in range(max_iters):
        grad = calculate_gradient(y, tx, w) + 2*lambda_*w # calcualte gradient
        w = w - gamma * grad # update the weights given the gradient and the step
        loss = calculate_loss(y, tx, w) + lambda_*np.linalg.norm(w) ** 2 # compute loss
        losses.append(loss) # append loss to the list of losses
        if iter % 25 == 0:
            gamma = gamma / 2 # decrease gamma after some iterations to increase the fitting of the parameters
            #print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return losses, w

##########################################
#         Cross Validation  &            #
#      Hyperparameter Finetuning         #
##########################################

# cross validation function for logistic regression
# k-indices = random subsets of the original samples
# k the set that we will use as the test set out of the subsets
# lambda_ = the lambda for the regularization function
# degree = the degree up to which we will exponentiate each feature
def cross_validation_logistic(y, x, k_indices, k, lambda_, degree, gamma = 3.0e-02):
    """return the loss of ridge regression."""
    N = y.shape[0] # the number of features
    k_fold = k_indices.shape[0]
    list_ = []
    interval = int(N/k_fold) # the length of each subset
    # this is not well written and should be changed it is very hard to understand
    # Create a list of the indices of subsets that are supposed to be used as a training set
    for i in range(k_fold):
        if i != k:
            list_.append(i)
    x_training = np.zeros((int((k_fold-1)/k_fold*N), x.shape[1]))
    y_training = np.zeros(int((k_fold-1)/k_fold*N))
    for j in range(len(list_)):
        x_training[interval*(j):interval*(j+1), :] = x[np.array([k_indices[list_[j]]]), :]
    for j in range(len(list_)):
        y_training[interval*(j):interval*(j+1)] = y[np.array([k_indices[list_[j]]])]
    # get the testing set out of the remaining set
    x_testing = x[k_indices[k], :]
    y_testing = y[k_indices[k]]
    # augment the testing and training set feature vectors
    x_training_augmented = build_poly(x_training, degree)
    x_testing_augmented = build_poly(x_testing, degree)
    #w_opt_training = ridge_regression(y_training, x_training_augmented, lambda_)
    # get optimal weights
    _,  w_opt_training = learning_by_penalized_gradient(y_training, x_training_augmented,
                                                        np.ones(x_training_augmented.shape[1]), gamma, 1000, lambda_)
    # calculate losses for the training and test set respectively and return them
    loss_tr = calculate_loss(y_training, x_training_augmented, w_opt_training)
    loss_te = calculate_loss(y_testing, x_testing_augmented, w_opt_training)
    return loss_tr, loss_te


# the following function aims at finetuning the hyperparameters for the logistic regression model
# tX = the array of features of the samples
# y = the label of each sample
# k_fold = the number of splits the dataset should be split into
# degrees = the range of the degrees to be tested for data augmentation
# lambdas = the different lambdas that can be used as a regularization param
def finetune_logistic(tX, y, gamma = 3.0e-02, k_fold=3, degrees = np.arange(2, 7), lambdas = np.logspace(-5,0,15)):
    seed = 1
    training_loss = np.zeros((len(lambdas), len(degrees)))# initial 2-d array for the grid search or the lambdas and the degrees
    testing_loss = np.zeros((len(lambdas), len(degrees)))# initial 2-d array for the grid search or the lambdas and the degrees
    k_indices = build_k_indices(y, k_fold, seed)#create the subarrays for the cross_validation
    for index1 in range(len(lambdas)):
        for index2 in range(len(degrees)):
            train_loss = 0 # initialize the training loss for each repetition
            test_loss = 0 # initialize the test loss for each repetition
            #run the cross validation for each possible split into test-train
            for k in range(k_fold):
                loss_tr, loss_te = cross_validation_logistic(y, tX, k_indices, k,
                                                    lambdas[index1], degrees[index2], gamma)
            train_loss += loss_tr # increase the training loss for this execution
            test_loss += loss_te # increase the test loss for this execution
            training_loss[index1, index2] = train_loss / k_fold # save the average of the training loss
            testing_loss[index1, index2] = test_loss / k_fold # save the average of the testing loss
    best_result = np.where(testing_loss == np.amin(testing_loss)) # get the optimal index for the hyper parameters
    # get and print the optimal values
    lambda_opt, degree_opt = lambdas[best_result[0]], degrees[best_result[1]]
    print(lambda_opt[0], degree_opt[0])
    return lambda_opt[0], degree_opt[0]

# calculate weights given the degree of data augmentation and the lambda_
def optimal_weights_logistic(tX,y,degree=2,lambda_=0):
    #Augment the feauture vector and calculate the optimal weights for logistic regression
    tX_augmented = build_poly(tX,degree)
    _, w_logistic_0 = learning_by_penalized_gradient(y, tX_augmented, np.zeros(tX_augmented.shape[1]), 3.0e-02,
                                              1000, lambda_= 1)

def predict_logistic(tX,w,degree=2):
    # make the predictions with the augmented test set
    #since we trained the model in augmented data, we augment the test set
    tX_augmented = build_poly(tX, degree)
    # make the predictions with the augmented test set and logistic regression
    predictions_logistic = sigmoid(tX_augmented @ w)
    return predictions_logistic
