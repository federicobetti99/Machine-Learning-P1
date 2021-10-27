# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from functools import partial


#def load_csv_data(data_path, sub_sample=False):
#    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
#    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
#    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
#    ids = x[:, 0].astype(np.int)
#    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
#    yb = np.ones(len(y))
#    yb[np.where(y=='b')] = -1

    # sub-sample
#    if sub_sample:
#        yb = yb[::50]
#        input_data = input_data[::50]
#        ids = ids[::50]

#    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

#compute accuracy of a prediction set
def compute_accuracy(y_test, pred):
    N = y_test.shape[0]
    accuracy = (y_test == pred).sum() / N
    return accuracy

# augment the feauture vector x by adding the powers of features
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=1 up to j=degree."""
    powers = np.arange(1, degree + 1)
    #phi = np.column_stack([np.power(x[:,0], exponent) for exponent in powers])
    phi = x[:,0]
    for i in range(1, x.shape[1]):
        phi_i = np.column_stack([np.power(x[:,i], exponent) for exponent in powers])
        phi = np.column_stack([phi, phi_i])
    return phi

#augments a row using polynomial and taking all covariance elements
def build_poly_cov_help(degree,x):
    res = [x]
    x_transp = x.T
    temp = x
    for i in range(degree-1):
        temp = temp * x_transp
        temp.flatten()
        res.append(temp)

    return res

def random_interval(low, high, size):
    sample = np.random.uniform(low, high, size)
    return sample

#augment feature vector X by raising to a certain polynomial the whole vector
# e.g. build_poly_cov([x1 x2]) = [x1 x1^2 x2 x2^2 x1x2]
def build_poly_cov(x,degree=2):
    x = np.apply_along_axis(partial(build_poly_cov_help,degree),1,x)
    return x


# build the k vector of shuffled indices
def build_k_indices(y, k_fold, seed):
    N = y.shape[0] # number of samples
    np.random.seed(seed) # initialize random seed
    interval = int(np.floor(N / k_fold)) # number of samples in each subset
    indices = np.random.permutation(N) # return an array of rnadomly order indices in [0,N)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)] # crete the sub arrays of indices in k-intervals
    return np.array(k_indices)
