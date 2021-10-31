import numpy as np
from proj1_helpers import *
from implementations import *

##############################################################
#            Ridge Regression                                #
##############################################################


#################################################
#       Cross Validation &                      #
#      Hyperparameter Finetuning                #
#################################################

# cross validation function for ridge regression
# k-indices = random subsets of the original samples
# k the set that we will use as the test set out of the subsets
# lambda_ = the lambda for the regularization function
# degree = the degree up to which we will exponentiate each feature

#for further clarifications please take a look at cross_validation_logistic
def cross_validation_ridge(y, x, k_indices, k, lambda_, degree,crossing = False):
    """
    The function performs cross validation selecting the best hyoerparameters for ridge regression namely lambda_
    (the regulatization coefficient) and degree (the degree used for feature expansion)
    param y: the labels
    param x: the training matrix
    param k_indices:  random subsets of indices of the original samples
    param k: the index such that k_indices[k] will be the indices of the set to use as the test set
    param lambda_: the coefficient of the regularization term
    param degree: the degree up to which we will exponentiate each feature (we are doing cross validation on this hyperparameter)
    param crossing: if crossing =True we will do crossing feature expansion of degree 2 (if we have x,y,z as the features
    we will expand as x,y,z,xy,xz,yz,x^2,y^2,z^2) if it is false we will expand using a polynomial expansion of degree=degree
    return: The accuracy computed for the given degree and lambda using k_indices[k] as the indices of test set
    """
    N = y.shape[0] # number of samples
    k_fold = k_indices.shape[0] # number of seperated sets
    list_ = []
    interval = int(N/k_fold) # the length of each subset
    # this is not well written and should be changed it is very hard to understand
    # Create a list of the indices of subsets that are supposed to be used as a training set
    for i in range(k_fold):
        if i != k:
            list_.append(i)
    # create the training set out of these indices
    x_training = np.zeros((int((k_fold-1)/k_fold*N), x.shape[1]))
    y_training = np.zeros(int((k_fold-1)/k_fold*N))
    column_to_add_train_0 = x_training[:, 0]
    for j in range(len(list_)):
        x_training[interval*(j):interval*(j+1), :] = x[np.array([k_indices[list_[j]]]), :]
    for j in range(len(list_)):
        y_training[interval*(j):interval*(j+1)] = y[np.array([k_indices[list_[j]]])]
    # get the testing set out of the remaining set
    x_testing = x[k_indices[k], :]
    y_testing = y[k_indices[k]]
    column_to_add_test_0 = x_testing[:, 0]
    # augment the testing and training set feature vectors
    if crossing is True:
        x_training_augmented = build_poly_cov(x_training[:,1:], degree)
        x_testing_augmented = build_poly_cov(x_testing[:,1:], degree)
        x_training_augmented = np.insert(x_training_augmented, 0, column_to_add_train_0, axis=1)
        x_testing_augmented = np.insert(x_testing_augmented, 0, column_to_add_test_0, axis=1)
    else:
        x_training_augmented = build_poly(x_training, degree)
        x_testing_augmented = build_poly(x_testing, degree)
    # get optimal weights
    w_opt_training,_ = ridge_regression(y_training, x_training_augmented, lambda_)
    # calculate accuracy for the test set and return it
    predictions_test = x_testing_augmented @ w_opt_training
    predictions_test = np.array([0 if el < 0.5 else 1 for el in predictions_test])
    acc_test = compute_accuracy(y_testing, predictions_test)
    return acc_test

# the following function aims at finetuning the hyperparameters of the ridge regression model
# tX = the array of features of the samples
# y = the label of each sample
# k_fold = the number of splits the dataset should be split into
# degrees = the range of the degrees to be tested for data augmentation
# lambdas = the different lambdas that can be used as a regularization param

def finetune_ridge(tX, y, k_fold = 5, degrees = np.arange(2, 7), lambdas = np.logspace(-5,0,15), crossing = False):
    """
    The function tunes the best hyperparameters for the ridge regression model
    param tX: the feature training matrix
    param y: the labels
    k_fold = the number of fold for cross validation
    param degrees: the degrees up to which we will exponentiate each feature (we are doing cross validation on this hyperparameter)
    param lambdas: the coefficients of the regularization term (we are doing cross validation on this hyperparameter)
    param crossing: if crossing =True we will do crossing feature expansion of degree 2 (if we have x,y,z as the features
    we will expand as x,y,z,xy,xz,yz,x^2,y^2,z^2) if it is false we will expand using a polynomial expansion of degree=degree
    return: The hyperparameters lambda_opt, degree_opt which maximizes the validation accuracy
    """
    seed = 1 # initialise the seed for the randomizer
    testing_acc = np.zeros((len(lambdas), len(degrees)))# initial 2-d array for the search or the lambads and the degrees
    k_indices = build_k_indices(y, k_fold, seed) #create the subarrays for the cross_validation
    for index1 in range(len(lambdas)):
        for index2 in range(len(degrees)):
            current_sum_test = 0 # we will sum the test accuracies and we will average them to get the average test
            # accuracy
            #run the cross validation for each possible split into test-train
            for k in range(k_fold):
                current_test_acc = cross_validation_ridge(y, tX, k_indices, k,
                                                    lambdas[index1], degrees[index2], crossing)
                current_sum_test += current_test_acc # we increase the sum of the accuracy that we will later average
            testing_acc[index1, index2] = current_sum_test / k_fold # save the average of the test_accuracy
    best_result = np.where(testing_acc == np.amax(testing_acc)) # get the optimal index for the hyper parameters
    # get and print the optimal values
    lambda_opt, degree_opt = lambdas[best_result[0]],degrees[best_result[1]]
    print('the optimal parameters lambda, degree are respectively', lambda_opt, degree_opt)
    print('the corresponding accuracy is',testing_acc[best_result[0], best_result[1]])
    return lambda_opt[0], degree_opt[0]

# calculate weights given the degree of data augmentation and the lambda_
def optimal_weights_ridge(tX, y, degree, lambda_, crossing = False):
    """
    The function calculates the weights corresponding to the optimal hyperparameters of the ridge regression model
    param tX: the feature training matrix
    param y: the labels
    param degree: the degree up to which we will exponentiate each feature
    param lambda_: the coefficient of the regularization term
    param crossing: if crossing =True we will do crossing feature expansion of degree 2 (for example if we have x,y,z
    as the features we will expand as x,y,z,xy,xz,yz,x^2,y^2,z^2) if it is false we will expand using a
    polynomial expansion of degree=degree
    return w_ridge: the best weights for ridge regression model
    """
    if crossing is False:
        tX_augmented = build_poly(tX, degree) #if it is false we do polynomial feature expansion
    if crossing is True: #if it is true we do crossing feature expansion
        column_to_add = tX[:,0] #we will expand everything a part from the column of ones (bias column)
        tX_augmented = build_poly_cov(tX, degree)
        tX_augmented = np.insert(tX_augmented, 0, column_to_add, axis=1) #we re-add the column of the bias term
    w_ridge,_ = ridge_regression(y, tX_augmented, lambda_)
    return w_ridge

def predict_ridge(tX, w, degree, crossing = False):
    """
    The function calculates the predictions using the ridge regression model with weihts w
    param tX: the test matrix
    param w: the weights of the ridge regression model
    param degree: the degree up to which we will exponentiate each feature
    param crossing: if crossing =True we will do crossing feature expansion of degree 2 (for example if we have x,y,z
    as the features we will expand as x,y,z,xy,xz,yz,x^2,y^2,z^2) if it is false we will expand using a
    polynomial expansion of degree=degree
    return predictions_ridge: the predictions for the ridge regression model
        """
    #since we trained the model in augmented data, we augment the test set
    if crossing is False:
        tX_augmented = build_poly(tX, degree) #if it is false we do polynomial feature expansion
    if crossing is True: #if it is true we do crossing feature expansion
        column_to_add = tX[:,0] #we will expand everything a part from the column of ones (bias column)
        tX_augmented = build_poly_cov(tX, degree)
        tX_augmented = np.insert(tX_augmented, 0, column_to_add, axis=1) #we re-add the column of the bias term
    # make the predictions with the augmented test set and ridge resgression
    predictions_ridge = tX_augmented @ w
    #we transform the predictions in -1,1
    predictions_ridge = np.array([-1 if el < 0.5 else 1 for el in predictions_ridge])
    return predictions_ridge
