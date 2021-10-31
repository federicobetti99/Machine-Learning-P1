import numpy as np
from proj1_helpers import *
from implementations import *


##############################################
#          Gradient Descent                  #
##############################################

def cross_validation_GD(y, x, k_indices, k, degree, gamma):
    """
    The function performs cross validation for gradient descent in order to set the best degree for feature augmentation
    :param y: the labels of the training set
    :param x: the feature matrix of the training set
    :param k_indices:  random subsets of indices of the original samples
    :param k: the index such that k_indices[k] will be the set to use as the test set out of the subsets
    :param degree: the degree up to which we will exponentiate each feature (we are doing cross validation on this hyperparameter)
    :param gamma: the learning rate
    :return: The accuracy computed for the given degree using k_indices[k] as the indices of test set
    """
    #first of all we construct the x_training,y_training,x_testing, y_testing
    N = y.shape[0]
    k_fold = k_indices.shape[0] #since k_indices is a list of lists his first dimension is the number of k-folds
    list_ = []
    interval = int(N/k_fold)
    for i in range(k_fold):
        if i != k:
            list_.append(i)
    x_training = np.zeros((int((k_fold-1)/k_fold*N), x.shape[1])) # we initialize x_training as the zero matrix with the
    #right dimensions
    y_training = np.zeros(int((k_fold-1)/k_fold*N)) #we do the same for y_training
    for j in range(len(list_)):
        x_training[interval*(j):interval*(j+1), :] = x[np.array([k_indices[list_[j]]]), :]
    x_testing = x[k_indices[k], :]
    for j in range(len(list_)):
        y_training[interval*(j):interval*(j+1)] = y[np.array([k_indices[list_[j]]])]
    y_testing = y[k_indices[k]]
    x_training_augmented = build_poly(x_training, degree)
    x_testing_augmented = build_poly(x_testing, degree)
    w_opt_training,_ = least_squares_GD(y_training, x_training_augmented, np.zeros(x_training_augmented.shape[1]) , 2000, gamma)
    predictions_test = x_testing_augmented@w_opt_training
    predictions_test = np.array([0 if el <0.5 else 1 for el in predictions_test])
    acc_test = compute_accuracy(y_testing, predictions_test)
    return acc_test

def finetune_GD(tX, y, k_fold = 4, degrees = np.arange(1,5)):
    """
    The function tunes the hyperparameter (the degree) which maximizes the test accuracy in the cross validation
    :param tX: the input matrix
    :param y: the labels
    :param k_fold:  the number of k-fold for cross validation
    :param degree: the degree up to which we will exponentiate each feature (we are doing cross validation on this hyperparameter)
    :return: The optimal degree
    """
    seed = 1
    testing_acc = np.zeros(len(degrees)) #it is a vector which will be filled, one component for each degree
    k_indices = build_k_indices(y, k_fold, seed)
    for index in range(len(degrees)):
        current_sum_test = 0 # we will sum the accuracy for each k in k_fold and we will average them at the end
        # in order to obtain the average accuracy correspondig to the current degree
        for k in range(k_fold):
            current_test_acc = cross_validation_GD(y, tX, k_indices, k, degrees[index], gamma = 5*10e-4)
            current_sum_test += current_test_acc # we sum the accuracies
        testing_acc[index] = current_sum_test / k_fold #we average them
    best_result = np.where(testing_acc == np.amax(testing_acc)) #we select the degree which maximises the accuracy
    degree_opt = degrees[best_result[0]]
    print('the degree which maximises the accuracy is', degree_opt)
    print('which correspond to an accuracy of', testing_acc[best_result[0]])
    return degree_opt

def optimal_weights_GD(tX,y,degree):
    """
    The function computes the optimal weights for gradient descent
    :param tX: the input matrix
    :param y: the labels
    :param degree: the degree up to which we will exponentiate each feature
    :return: w_opt_GD: the optimal weigths for the gradient descent
    """
    tX_augmented = build_poly(tX, degree) # we augment each feature
    #we use the function least_squares_GD to compute the optimal weights
    w_opt_GD,_ = least_squares_GD(y, tX_augmented, np.zeros(tX_augmented.shape[1]) , 2000, 5*10e-4) 
    return w_opt_GD

def predict_GD(tX,w,degree=2):
    """
    The function predicts the labels using the optimal weights of GD
    :param tX: the input matrix
    :param w: the optimal weghts of GD
    :param degree: the degree up to which we will exponentiate each feature
    :return: predictions_GD: the predictions made using GD 
    """
    # make the predictions with the augmented test set
    #since we trained the model in augmented data, we augment the test set
    tX_augmented = build_poly(tX, degree)
    # make the predictions with the augmented test set and GD
    predictions_GD = tX_augmented @ w
    #we transform the labels in -1,1
    predictions_GD = np.array([-1 if el < 0.5 else 1 for el in predictions_GD])
    return predictions_GD


###########################################################
#       Stochastic Gradient Descent                       #
###########################################################

def cross_validation_SGD(y, x, k_indices, k, degree, gamma):
    """
    The function performs cross validation for stochastic gradient descent in order to set the best degree for feature augmentation
    :param y: the labels of the training set
    :param x: the feature matrix of the training set
    :param k_indices:  random subsets of indices of the original samples
    :param k: the index such that k_indices[k] will be the set to use as the test set out of the subsets
    :param degree: the degree up to which we will exponentiate each feature (we are doing cross validation on this hyperparameter)
    :param gamma: the learning rate
    :return: The accuracy computed for the given degree using k_indices[k] as the indices of test set
    """
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
    w_opt_training,_ = least_squares_SGD(y_training, x_training_augmented, np.zeros(x_training_augmented.shape[1]) , 1000, gamma)
    predictions_test = x_testing_augmented@w_opt_training
    predictions_test = np.array([0 if el <0.5 else 1 for el in predictions_test])
    acc_test = compute_accuracy(y_testing, predictions_test)
    return acc_test

def finetune_SGD(tX, y, k_fold = 4, degrees = np.arange(1,5)):
    """
    The function tunes the hyperparameter (the degree) which maximizes the test accuracy in the cross validation
    :param tX: the input matrix
    :param y: the labels
    :param k_fold:  the number of k-fold for cross validation
    :param degree: the degree up to which we will exponentiate each feature (we are doing cross validation on this hyperparameter)
    :return: The optimal degree
    """
    seed = 1
    testing_acc = np.zeros(len(degrees)) #it is a vector which will be filled, one component for each degree
    k_indices = build_k_indices(y, k_fold, seed)
    for index in range(len(degrees)):
        current_sum_test = 0 # we will sum the accuracy for each k in k_fold and we will average them at the end
        # in order to obtain the average accuracy correspondig to the current degree
        for k in range(k_fold):
            current_test_acc = cross_validation_SGD(y, tX, k_indices, k, degrees[index], gamma = 5*10e-4)
            current_sum_test += current_test_acc # we sum the accuracies
        testing_acc[index] = current_sum_test / k_fold # we average them
    best_result = np.where(testing_acc == np.amax(testing_acc)) # we select the degree which maximises the accuracy
    degree_opt = degrees[best_result[0]]
    print('the degree which maximises the accuracy is', degree_opt)
    print('which correspond to an accuracy of', testing_acc[best_result[0]])
    return degree_opt

def optimal_weights_SGD(tX, y, degree):
    """
    The function computes the optimal weights for stochastic gradient descent
    :param tX: the input matrix
    :param y: the labels
    :param degree: the degree up to which we will exponentiate each feature
    :return: w_opt_SGD: the optimal weigths for the stochastic gradient descent
    """
    tX_augmented = build_poly(tX, degree) # we augment the features for the test set
    w_opt_SGD,_ = least_squares_SGD(y, tX_augmented, np.zeros(tX_augmented.shape[1]) , 2000, 5*10e-4)
    return w_opt_SGD

def predict_SGD(tX, w, degree=2):
    """
    The function predicts the labels using the optimal weights of SGD
    param tX: the input matrix
    param w: the optimal weghts of SGD
    param degree: the degree up to which we will exponentiate each feature
    return: predictions_SGD: the predictions made using SGD
    """
    # make the predictions with the augmented test set
    #since we trained the model in augmented data, we augment the test set
    tX_augmented = build_poly(tX, degree)
    # make the predictions with the augmented test set and GD
    predictions_SGD = tX_augmented @ w
    # we transform the labels in -1,1
    predictions_GD = np.array([-1 if el < 0.5 else 1 for el in predictions_GD])
    return predictions_SGD