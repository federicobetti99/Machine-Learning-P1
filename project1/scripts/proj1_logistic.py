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
def cross_validation_logistic(y, x, k_indices, k, lambda_, degree, gamma):
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
    _, w_opt_training = learning_by_penalized_gradient(y_training, x_training_augmented,
                                                       np.zeros(x_training_augmented.shape[1]), gamma, 1000, lambda_)
    predictions_test = sigmoid(x_testing_augmented @ w_opt_training)
    predictions_test = np.array([0 if el < 0.5 else 1 for el in predictions_test])
    acc_test = compute_accuracy(y_testing, predictions_test)
    return acc_test


# the following function aims at finetuning the hyperparameters for the logistic regression model
# tX = the array of features of the samples
# y = the label of each sample
# k_fold = the number of splits the dataset should be split into
# degrees = the range of the degrees to be tested for data augmentation
# lambdas = the different lambdas that can be used as a regularization param
def finetune_logistic(tX, y, gamma , degrees, lambdas, k_fold=4):
    seed = 1
    testing_acc = np.zeros((len(lambdas), len(degrees)))
    k_indices = build_k_indices(y, k_fold, seed)
    for index1 in range(len(lambdas)):
        for index2 in range(len(degrees)):
            test_acc = 0
            for k in range(k_fold):
                current_test_acc = cross_validation_logistic(y, tX, 
                                                            k_indices, k, lambdas[index1], degrees[index2], gamma)
                test_acc += current_test_acc
            testing_acc[index1, index2] = test_acc / k_fold
    best_result = np.where(testing_acc == np.amax(testing_acc))
    lambda_opt, degree_opt = lambdas[best_result[0]], degrees[best_result[1]]
    print(lambda_opt, degree_opt)
    print(np.amax(testing_acc))
    return lambda_opt[0], degree_opt[0]

# calculate weights given the degree of data augmentation and the lambda_
def optimal_weights_logistic(tX, y, gamma, degree, lambda_):
    #Augment the feauture vector and calculate the optimal weights for logistic regression
    tX_augmented = build_poly(tX,degree)
    _, w_logistic = learning_by_penalized_gradient(y, tX_augmented, np.zeros(tX_augmented.shape[1]), gamma,
                                              1000, lambda_= 1)
    return w_logistic

def predict_logistic(tX, w, degree):
    # make the predictions with the augmented test set
    #since we trained the model in augmented data, we augment the test set
    tX_augmented = build_poly(tX, degree)
    # make the predictions with the augmented test set and logistic regression
    predictions_logistic = sigmoid(tX_augmented @ w)
    return predictions_logistic

# calculate the batch gradient for logistic regression
def calculate_batch_gradient(y, tx, w, batchsize):
    random_indices = np.random.randint(0, y.shape[0], batchsize)
    tx_small_rand = tx[random_indices]
    y_small_rand = y[random_indices]
    w_small_rand = w[random_indices]
    return np.transpose(tx_small_rand) @ (sigmoid(tx_small_rand @ w_small_rand) - y_small_rand), random_indices

def learning_by_penalized_batch_gradient(y, tx, w_initial, gamma, max_iters, lambda_, batchsize):
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
        grad = calculate_batch_gradient(y, tx, w, batchsize=1000)
        grad = grad + 2*lambda_*w[random_indices]
        w = w - gamma * grad
        loss = calculate_loss(y, tx, w) + lambda_*np.linalg.norm(w) ** 2
        losses.append(loss)
        if iter % iter_drop == 0:
            gamma = gamma * drop ** np.floor((1+iter) / (iter_drop))
            #print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return losses, w

def cross_validation_logistic_batch(y, x, k_indices, k, lambda_, degree, gamma):
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
    _, w_opt_training = learning_by_penalized_batch_gradient(y_training, x_training_augmented,
                                                       np.zeros(x_training_augmented.shape[1]), gamma, 7000, lambda_, 1000)
    predictions_test = sigmoid(x_testing_augmented @ w_opt_training)
    predictions_test = np.array([0 if el < 0.5 else 1 for el in predictions_test])
    acc_test = compute_accuracy(y_testing, predictions_test)
    return acc_test

def finetune_batch_logistic(tX, y, gamma , degrees, lambdas, k_fold=4):
    seed = 1
    testing_acc = np.zeros((len(lambdas), len(degrees)))
    k_indices = build_k_indices(y, k_fold, seed)
    for index1 in range(len(lambdas)):
        for index2 in range(len(degrees)):
            test_acc = 0
            for k in range(k_fold):
                current_test_acc = cross_validation_logistic_batch(y, tX, 
                                                            k_indices, k, lambdas[index1], degrees[index2], gamma)
                test_acc += current_test_acc
            testing_acc[index1, index2] = test_acc / k_fold
    best_result = np.where(testing_acc == np.amax(testing_acc))
    lambda_opt, degree_opt = lambdas[best_result[0]], degrees[best_result[1]]
    print(lambda_opt, degree_opt)
    print(np.amax(testing_acc))
    return lambda_opt[0], degree_opt[0]