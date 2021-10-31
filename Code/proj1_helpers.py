"""some helper functions for project 1."""
import csv
import numpy as np
from functools import partial


def load_csv_data(data_path, sub_sample=False):
    """
     Loads data and returns y (class labels), tX (features) and ids (event ids)
    :param data_path: the path of the file of interest
    :param sub_sample: takes only a subset of the data contained in data_path
    :return: the output vector, the dataset matrix, the ids of the events of interest
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def ensemble_predictions(predictions_1, predictions_2, predictions_3):
    """
    Returns a unique predictions given three different predictions by picking the majority for each element of the
    three lists, namely -1 if predictions_1[i] + predictions_2[i] + predictions_3[i] < 1 (it means there are at least
    two -1's, 1 otherwise
    :param predictions_1: the predictions coming from the first model
    :param predictions_2: the predictions coming from the second model
    :param predictions_3: the predictions coming from the third model
    :return:
    """
    average_predictions = predictions_1 + predictions_2 + predictions_3
    average_predictions = np.array([1 if el >= 1 else -1 for el in average_predictions])
    return average_predictions


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    :param ids: event ids associated with each prediction
    :param y_pred:  predicted class labels
    :param name: string name of .csv output file to be created
    :return: 0
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def standardize(x):
    """
    Standardize the original data set.
    :param x: the dataset to standardize
    :return: the dataset standardized with mean 0 and variance 1
    """
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def standardize_test(x, mean, std):
    """
    Standardize the test set using the mean and standard deviation of the clusters of the training.
    :param x: the dataset matrix to be standardized in each entry
    :param mean: the mean of the training corresponding cluster
    :param std: the standard deviation of the training corresponding cluster
    :return: the test dataset standardized with mean 0 and variance 1
    """
    x = x - mean
    x = x / std
    return x

##############################################
#       Loss + Gradient Calculators          #
#       Linear Model                         #
##############################################


def compute_loss(y, tX, w):
    """
    Returns the loss calculated using Mean Squared Error
    :param y: the vector of the outputs
    :param tX: the dataset matrix
    :param w: the weights vector
    :return: the MSE loss calculated in w
    """
    N = y.shape[0]
    e = y - tX @ w
    loss = 1/(2*N) * np.dot(e, e)
    return loss


def compute_gradient(y, tX, w):
    """
     Returns the gradient of the Mean Squared Error loss function as a function of the weights
    :param y: the vector of outputs
    :param tX: the dataset matrix
    :param w: the weights vector
    :return: the gradient of the MSE loss function calculated in w
    """
    N = y.shape[0]
    e = y - tX @ w
    gradient = -(1/N) * (tx.T) @ (e)
    return gradient


def compute_stoch_gradient(y, tX, w):
    """
    Returns a random component of the gradient vector of the MSE loss function as a function of the weights
    :param y: the vector of the outputs
    :param tX: the dataset matrix
    :param w: the weights vector
    :return: a random component of the gradient vector of the MSE loss function calculated in w
    """
    N = y.shape[0]
    # pick a random index in the interval [0, N-1]
    random_number = np.random.randint(0,N)
    # get the sample of that index
    xn = tX[random_number, :]
    # calculate the stochastic gradient as the random_number-th component of the gradient vector
    random_gradient = - np.dot(xn, y[random_number] - np.dot(xn,w))
    return random_gradient

###############################################
#        sigmoid                              #
###############################################


def sigmoid(t):
    """
    This function calculates the sigmoid function for a vector t
    :param t: the vector to evaluate the sigmoid function at
    :return: the image of t under the sigmoid or logistic function
    """
    # separating the positive and the negative components of the vector t
    positive_indices = np.where(t >= 0)[0]
    negative_indices = np.where(t < 0)[0]
    # the sigmoid function is then calculated depending on the sign of each component:
    # if t[i] >= 0 we calculate it as 1 / (1+exp(-t[i]))
    # if t[i] < 0 we calculate it as exp(t[i]) / (1+exp(t[i])
    # we then return the vector with the components calculated in such a way
    z = np.zeros(len(t))
    z[positive_indices] = 1 / (1+np.exp(-t[positive_indices]))
    z[negative_indices] = np.exp(t[negative_indices]) / (1 + np.exp(t[negative_indices]))
    return z

##############################################
#        calculate loss                      #
#        calculate gradient                  #
##############################################


def calculate_loss(y, tX, w):
    """
    This function calculates the loss of the logistic function which is again calculated by manipulating the formal definition in order
    to avoid overflows and numerical problems
    :param y: the vector of the outputs
    :param tX: the dataset matrix
    :param w: the weights vector
    :return: the loss associated to the logistic function calculated in w
    """
    # separating the positive and the negative components of the vector tX @ w
    pos_ind = np.where(tX @ w >= 0)[0]
    neg_ind = np.where(tX @ w < 0)[0]
    # we then calculate the loss in two different ways depending on the sign of each component:
    # if z[i] = (tX @ w)[i] >= 0 we rewrite log(1+exp(z[i]) as z[i] + log(1+exp(-z[i]))
    # if z[i] = (tX @ w)[i] < 0 we rewrite log(1+exp(z[i]) as -z[i] + log(1+exp(z[i]))
    # we then return the sum of the sums of the two vectors components which determines the total loss
    loss_pos = - y[pos_ind] * (tX @ w)[pos_ind] + (tX @ w)[pos_ind] + np.log(1+np.exp(-(tX @ w)[pos_ind]))
    loss_neg = - y[neg_ind] * (tX @ w)[neg_ind] - (tX @ w)[neg_ind] + np.log(1+np.exp((tX @ w)[neg_ind]))
    return loss_pos.sum() + loss_neg.sum()


def calculate_gradient(y, tX, w):
    """
    Returns the gradient for logistic regression. Note that this calculation is not suffering any numerical problems
    because of how sigmoid (tX @ w) is calculated from the above
    :param y: the vector of the outputs
    :param tX: the dataset matrix
    :param w: the weights vector
    :return: the gradient of the loss function associated to the logistic regression calculated in w
    """
    return np.transpose(tX) @ (sigmoid(tX @ w) - y)


def compute_accuracy(y_test, pred):
    """
    compute accuracy of a prediction set
    y_test:
    pred: the predictions
    :param y_test: the output to be compared with the predictions
    :param pred: the predictions
    :return: the number of correct predictions, i.e. the percentage of elements for which y_test and pred coincide
    """
    N = y_test.shape[0]
    accuracy = (y_test == pred).sum() / N
    return accuracy


def build_poly(x, degree):
    """
    This functions augments without crossing the feature matrix x adding the powers of features
    by means of a polynomial expansion for j=1 up to j=degree.
    :param x: the dataset of the features to be augmented
    :param degree: the maximum degree of expansion of each feature
    :return: the new matrix where each column outside of the bias term has been expanded polynomially
    """
    powers = np.arange(1, degree + 1)
    # the bias term is not expanded as its expansion is not enriching the model and only creates conditioning problems
    phi = x[:, 0]
    for i in range(1, x.shape[1]):
        # at each iteration the expansion of each column is calculated
        phi_i = np.column_stack([np.power(x[:, i], exponent) for exponent in powers])
        # the expansion of the i-th feature is stacked to the end of the already existing matrix phi
        phi = np.column_stack([phi, phi_i])
    return phi


def random_interval(low, high, size):
    """
    Returns a sample of values for randomized grid search in the interval [low, high]
    low: lower bound of the interval of grid search of the optimal hyper-parameters
    high: upper bound of the interval of grid search of the optimal hyper-parameters
    size: size of the sample
    """
    sample = np.random.uniform(low, high, size)
    return sample


def build_k_indices(y, k_fold, seed):
    """
    This function constructs the indices to be used in cross-validation
    :param y: the output vector
    :param k_fold: the number of subsets one wants to split each feature into
    :param seed: the random seed used to perform the splitting
    :return: an array containing the shuffled indices between 0 and N
    """
    N = y.shape[0] # number of samples
    np.random.seed(seed) # initialize random seed
    interval = int(np.floor(N / k_fold)) # number of samples in each subset
    indices = np.random.permutation(N) # return an array of rnadomly order indices in [0,N)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)] # crete the sub arrays of indices in k-intervals
    return np.array(k_indices)


def build_poly_cov_help(degree, x):
    """
    This function is a helper to construct the crossing feature expansion
    :param degree: the degree of the expansion
    :param x: the dataset in which we are performing the augmentation of the data
    :return: the terms to be added for the crossing expansion
    """
    res = x
    N = x.shape[0]
    x_transp = res[:, None]
    temp = res
    temp = x_transp * temp
    nt = []
    count = 0
    for i in range(N):
        nt.append(temp[i,count:].tolist())
        count = count + 1
    fl = [i for item in nt for i in item]
    res = x.tolist() + fl
    return res


def build_poly_cov(x, degree = 2):
    """
    This functions returns the crossing feature expansion of x
    :param x: the dataset
    :param degree: set to 2 by default as this was the better performing degree
    :return: the augmented matrix with all the polynomial expansions and the covariance terms
    """
    x = np.apply_along_axis(partial(build_poly_cov_help, degree), 1, x)
    return x
