import numpy as np
from proj1_helpers import *
from implementations import *

##########################################
#         Cross Validation  &            #
#      Hyper-parameter Finetuning         #
##########################################

def cross_validation_logistic(y, x, k_indices, k, lambda_, degree, gamma, crossing = False):
    """
    cross-validation function for logistic regression
    :param y: the vector of the outputs for training the models
    :param x: the dataset matrix
    :param k_indices: random shuffling of the original samples
    :param k: index for the choice of the testing subset, ranging from 1 up to k-fold
    :param lambda_: the regularization term
    :param degree: the degree for augmentation of each feature
    :param gamma: the learning rate for penalized gradient descent
    :param crossing: a boolean which is True if we perform crossing feature expansion, False otherwise
    :return: the testing accuracy calculated for the hyper-parameters given in input
    """
    N = y.shape[0]
    k_fold = k_indices.shape[0]
    list_ = []
    # we calculate the length of each sub-interval as N/k_fold casted
    interval = int(np.floor(N/k_fold))
    # getting all the indices which are not the k-th used for testing
    for i in range(k_fold):
        if i != k:
            list_.append(i)
    x_training = np.zeros((int((k_fold-1)/k_fold*N), x.shape[1]))
    y_training = np.zeros(int((k_fold-1)/k_fold*N))
    # we extract the bias term as if crossing = True it is a feature we don't want to expand
    column_to_add_train_0 = x_training[:, 0]
    # adding to the training subset all the elements of x which are not in the k-th subset
    for j in range(len(list_)):
        x_training[interval*(j):interval*(j+1), :] = x[np.array([k_indices[list_[j]]]), :]
    # adding to the testing subset only the elements of the k-th subset
    x_testing = x[k_indices[k], :]
    # we extract the bias term as if crossing = True it is a feature we don't want to expand
    column_to_add_test_0 = x_testing[:, 0]
    # we perform the same operations as above also on the output vector y
    for j in range(len(list_)):
        y_training[interval*(j):interval*(j+1)] = y[np.array([k_indices[list_[j]]])]
    y_testing = y[k_indices[k]]
    if crossing is True:
        # we perform crossing feature expansion on the training and testing submatrices outside of the bias term,
        # which is added manually after this operation
        x_training_augmented = build_poly_cov(x_training[:,1:], degree)
        x_testing_augmented = build_poly_cov(x_testing[:,1:], degree)
        x_training_augmented = np.insert(x_training_augmented, 0, column_to_add_train_0, axis=1)
        x_testing_augmented = np.insert(x_testing_augmented, 0, column_to_add_test_0, axis=1)
    else:
        # we perform the standard augmentation of each feature without crossing
        x_training_augmented = build_poly(x_training, degree)
        x_testing_augmented = build_poly(x_testing, degree)
    # we compute by iteration the vector of the optimal weights
    w_opt_training, _ = reg_logistic_regression(y_training, x_training_augmented, lambda_,
                                                       np.zeros(x_training_augmented.shape[1]), 1000, gamma)
    # we do the predictions on the testing submatrix in {0,1}
    predictions_test = sigmoid(x_testing_augmented @ w_opt_training)
    predictions_test = np.array([0 if el < 0.5 else 1 for el in predictions_test])
    # we compute the accuracy of the prediction with respect to the output testing vector
    acc_test = compute_accuracy(y_testing, predictions_test)
    return acc_test

def finetune_logistic(tX, y, gamma , degrees, lambdas, k_fold=4, crossing = False):
    """
    This function aims at finetuning the hyper-parameters for the logistic regression model
    :param tX: the array of features of the samples
    :param y: the label of each sample
    :param gamma: the learning rate
    :param degrees: the range of the degrees to be tested for data augmentation
    :param lambdas: the different lambdas that can be used as a regularization param
    :param k_fold: the number of splits the dataset should be divided into
    :param crossing:  a boolean which is True if we perform crossing feature expansion, False otherwise
    :return: the degree and lambda which are maximizing the test accuracy calculated by the cross-validation function
    """
    seed = 3
    testing_acc = np.zeros((len(lambdas), len(degrees)))
    k_indices = build_k_indices(y, k_fold, seed)
    for index1 in range(len(lambdas)):
        for index2 in range(len(degrees)):
            test_acc = 0
            for k in range(k_fold):
                current_test_acc = cross_validation_logistic(y, tX,
                                                            k_indices, k, lambdas[index1], degrees[index2], gamma, crossing)
                test_acc += current_test_acc
            # we average the k-fold results obtained above and we store the obtained value in the testing_acc matrix
            testing_acc[index1, index2] = test_acc / k_fold
    # we then pick the entry of the matrix corresponding to the highest accuracy on the testing subset
    best_result = np.where(testing_acc == np.amax(testing_acc))
    # the optimal lambda and degree for our logistic regression model are recovered and returned by the function
    lambda_opt, degree_opt = lambdas[best_result[0]], degrees[best_result[1]]
    # this print are only for visualization of the best accuracy for each submodel in the cluster
    print("The Optimal λ is:", lambda_opt[0])
    print("The Optimal Augmentation Degree is:",degree_opt[0])
    print("The Accuracy is the Validation Set for these parameters is:",np.amax(testing_acc))
    return lambda_opt[0], degree_opt[0]

def optimal_weights_logistic(tX, y, gamma, degree, lambda_, crossing = False):
    """
    This function calculates the optimal weights for the logistic regression given the hyperparameters found previously
    :param tX: the array of features of the samples
    :param y: the label of each sample
    :param gamma: the learning rate
    :param degree: the optimal degree for data augmentation
    :param lambda_: the optimal regularization parameter (simply 0 if we perform logistic regression without regularization)
    :param crossing: a boolean which is True if we perform crossing feature expansion, False otherwise
    :return: the optimal weights calculated by regularized logistic regression with the hyper-parameters returned by the finetuning
    """
    if crossing is False:
        # we perform the standard polynomial expansion of each feature
        tX_augmented = build_poly(tX, degree)
    if crossing is True:
        # we extract the bias term
        column_to_add = tX[:,0]
        # we perform crossing feature expansion on the rest of the columns and we add manually the bias term again
        tX_augmented = build_poly_cov(tX, degree)
        tX_augmented = np.insert(tX_augmented, 0, column_to_add, axis=1)
    # we calculate the optimal weights by penalized gradient descent
    w_logistic, _ = reg_logistic_regression(y, tX_augmented, lambda_, np.zeros(tX_augmented.shape[1]), 1000, gamma)
    return w_logistic

def predict_logistic(tX, w, degree, crossing = False):
    """
    This function calculates the predictions for the logistic regression model
    :param tX: the array of features of the samples
    :param w: the optimal weights for the model
    :param degree: the optimal degree for polynomial augmentation of each feature
    :param crossing: a boolean which is True if we perform crossing feature expansion, False otherwise
    :return: the logistic regression predictions in {-1,1}
    """
    #since we trained the model in augmented data, we augment the test set
    if crossing is False:
        # we perform the standard polynomial expansion of each feature
        tX_augmented = build_poly(tX, degree)
    if crossing is True:
        # we extract the bias term
        column_to_add = tX[:,0]
        # we perform crossing feature expansion on the rest of the columns and we add manually the bias term again
        tX_augmented = build_poly_cov(tX, degree)
        tX_augmented = np.insert(tX_augmented, 0, column_to_add, axis=1)
    # we make the predictions with the augmented test set
    predictions_logistic = sigmoid(tX_augmented @ w)
    predictions_logistic = np.array([-1 if el < 0.5 else 1 for el in predictions_logistic])
    return predictions_logistic

def reg_logistic_regression_plot(y_training, y_testing, tx_training, tx_testing, lambda_, w_initial, gamma, max_iters):
    """
    The main use of this function is to return a slightly altered result to facilitate the plots
    :param y: the output of vectors
    :param tx: the dataset matrix
    :param w_initial: the initial weights to start the descent algorithm
    :param gamma: the learning rate or step size
    :param max_iters: the maximum number of iterations
    :param lambda_: the regularization parameter
    :return: the weights at the iteration max_iters or when the algorithm stops by convergence criterion and the set of training losses
    """
    threshold = 1e-8
    train_losses = []
    test_losses = []
    drop = 0.5
    iter_drop = 25
    w = w_initial
    for iter in range(max_iters):
        grad = calculate_gradient(y_training, tx_training, w) + 2 * lambda_ * w
        w = w - gamma * grad
        train_loss = calculate_loss(y_training, tx_training, w) + lambda_ * np.linalg.norm(w) ** 2
        testing_loss = calculate_loss(y_testing, tx_testing, w) + lambda_ * np.linalg.norm(w) ** 2
        train_losses.append(train_loss)
        test_losses.append(testing_loss)
        if iter % iter_drop == 0:
            gamma = gamma * drop ** np.floor((1+iter) / (iter_drop))
            #print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        if len(train_losses) > 1 and np.abs(train_losses[-1] - train_losses[-2]) < threshold:
            break
    return w, train_losses,test_losses

# calculate the batch gradient for logistic regression
def calculate_batch_gradient(y, tx, w, batchsize):
    """
    This function returns the gradient calculated only in batchsize random components
    :param y: the vector of the outputs
    :param tx: the array of features of the samples
    :param w: the optimal weights for the model
    :param batchsize: the size of the batch
    :return: batcshize random components of the gradient of the logistic loss calculated in w
    """
    random_indices = np.random.randint(0, y.shape[0], batchsize)
    tx_small_rand = tx[random_indices]
    y_small_rand = y[random_indices]
    w_small_rand = w[random_indices]
    return np.transpose(tx_small_rand) @ (sigmoid(tx_small_rand @ w_small_rand) - y_small_rand), random_indices

def learning_by_penalized_batch_gradient(y, tx, w_initial, gamma, max_iters, lambda_, batchsize):
    """
    This function performs stochastic gradient descent using as a cost function to minimize the logistic cost
    :param y: the vector of the outputs
    :param tx: the array of features of the samples
    :param w_initial: initial weights for the descent algorithm
    :param gamma: the learning rate
    :param max_iters: maximum number of iterations
    :param lambda_: the regularization parameter
    :param batchsize: the size of the batch
    :return: the final weights and the final loss of the iterative algorithm of regularized logistic regression
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
    final_loss = losses[-1]
    return w, final_loss

#for further clarifications please take a look at cross_validation_logistic
def cross_validation_logistic_batch(y, x, k_indices, k, lambda_, degree, gamma):
    """
    This function performs the cross validation on the stochastic gradient descent for the regularized logistic regression
    :param y: the vector of the outputs
    :param x: the array of features of the samples
    :param k_indices: random shuffling of the original samples
    :param k: index for the testing subset, ranging from 1 up to k-fold
    :param lambda_: the regularization parameter
    :param degree: the degree for augmentation of each feature
    :param gamma: the learning rate for penalized gradient descent
    :return: the testing accuracy calculated with the hyper-parameters given in input
    """
    N = y.shape[0]
    k_fold = k_indices.shape[0]
    list_ = []
    interval = int(np.floor(N/k_fold))
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
    """
     This function aims at finetuning the hyper-parameters for the logistic regression model
    :param tX: the array of features of the samples
    :param y: the label of each sample
    :param gamma: the learning rate
    :param degrees: the range of the degrees to be tested for data augmentation
    :param lambdas: the different lambdas tested as a regularization parameter
    :param k_fold: the number of splits the dataset should be divided into
    :return: the degree and lambda values for which the testing accuracy is maximized
    """
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

#############################################
#        newton method                      #
#############################################
############################################
# The following functions are not part of  #
# any implementation                       #
############################################

def calculate_hessian(y, tX, w):
    """
    This function returns the Hessian of the logistic loss function: it is a well known fact that the Hessian can be
    factorized as tX.T @ D @ tX where D is a diagonal matrix whose i-th diagonal element is the derivative of the
    logistic function evaluated in (tX @ w)[i]
    :param y: the vector of the outputs
    :param tX: the dataset matrix
    :param w: the weights vector
    :return: the Hessian of the loss associated to the logistic function calculated in w
    """
    diag = sigmoid(tX @ w) * (1 - sigmoid(tX @ w))
    D = diag * np.eye(tX.shape[0])
    return np.transpose(tX) @ D @ tX


def logistic_regression_compute(y, tX, w):
    """
    This function calculates the main quantities used in the logistic regression framework
    :param y: the vector of the outputs
    :param tX: the dataset matrix
    :param w: the weights vector
    :return: the loss, the gradient and the Hessian calculated in the previous functions above
    """
    grad = calculate_gradient(y, tX, w)
    hess = calculate_hessian(y, tX, w)
    loss = calculate_loss(y, tX, w)
    return loss, grad, hess

def learning_by_newton_method(y, tx, w, gamma):
    """
    This function implements the Newton Method, i.e. the standard gradient descent is modified as the
    iterative descent algorithm now follows the direction of the gradient subject to the affine transformation
    performed by the second order geometry of the system
    :param y: the vector of the outputs
    :param tx: the dataset matrix
    :param w: the weights vector
    :param gamma: the learning rate
    :return: the loss and the weights calculated after one iteration of the Newton's Method for the logistic regression
    """
    loss, grad, hess = logistic_regression_compute(y, tx, w)
    sol = np.linalg.solve(hess, grad)
    w = w - gamma * sol
    return w, loss

############################################
#  penalized logistic regression           #
############################################
############################################
# The following function is not part of    #
# any implementation                       #
############################################


def penalized_logistic_regression(y, tx, w, lambda_):
    """
    This function calculates the main quantities used in the regularized logistic regression framework
    :param y: the vector of the outputs
    :param tx: the dataset matrix
    :param w: the weights vector
    :param lambda_: the regularization parameter
    :return: the gradient,
    """
    loss = calculate_loss(y, tx, w) + lambda_ * np.linalg.norm(w) ** 2
    grad = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    hess = calculate_hessian(y, tx, w) + 2 * lambda_ * np.eye(w.shape[0])
    return loss, grad, hess
