import numpy as np
from proj1_helpers import *


def load_csv_data(data_path, sub_sample=False):
    """
    Loads data and returns y (class labels), tX (features) and ids (event ids)
    :param data_path: path of the data (of the csv file
    :param sub_sample: True if we want to load a subsample of the data, false if we wnt to load the whole file
    :return: y (class labels), tX (features) and ids (event ids)
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_output(tX_test, predictions_0, predictions_1, predictions_2, predictions_3):
    """
    The function creates the final predictions i.e., given the predictions associated to each
    submatrix, the function stacks together the predictions constructing a single vector of labels
    (-1,1)
    :param tX_test: The test matrix
    :param predictions_0: The predictions associated to tX_tilda_0
    :param predictions_1: The predictions associated to tX_tilda_1
    :param predictions_2: The predictions associated to tX_tilda_2
    :param predictions_3: The predictions associated to tX_tilda_3
    :return: The final predictions (a vector of labels (-1,1))
    """
    # we construct a vector of the right dimension for the final predictions
    stacked_predictions = np.zeros(tX_test.shape[0])
    # we find the indices which have 0,1,2 or 3 Pri-Jets number
    zero_indices, one_indices, two_indices, three_indices = split_to_Jet_Num_Help(tX_test)
    # we insert in the vector of the final predictions the predictions for each submatrix in the right place
    stacked_predictions[zero_indices] = predictions_0
    stacked_predictions[one_indices] = predictions_1
    stacked_predictions[two_indices] = predictions_2
    stacked_predictions[three_indices] = predictions_3
    return stacked_predictions


def split_to_Jet_Num_Help(tX):
    """
    This functions return the indices corresponding to the rows of tX where the Pri-jet-number is equal to
    zero, one, two or three respectively
    :param tX: The training or testing matrix
    :return: zero_indices, one_indices, two_indices, three_indices
    """
    zero_indices = []
    one_indices = []
    two_indices = []
    three_indices = []
    zero_indices = np.where(tX[:, 22] == 0)[0]  # we select the indices such that they have 0 as pri-jet-num
    one_indices = np.where(tX[:, 22] == 1)[0]  # we select the indices such that they have 1 as pri-jet-num
    two_indices = np.where(tX[:, 22] == 2)[0]  # we select the indices such that they have 2 as pri-jet-num
    three_indices = np.where(tX[:, 22] == 3)[0]  # we select the indices such that they have 3 as pri-jet-num
    return zero_indices, one_indices, two_indices, three_indices


def split_to_Jet_Num(tX):
    """
    The function create the four submatrices from the big initial matrix tX. Each of the submarix contains only the
    rows of tX such that the pri-jet-num is equal to zero, one, two or three respectively
    :param tX: The input matrix (it could be training or testing)
    :return: tX_0,tX_1,tX_2,tX_3 the three submatrices
    """
    # we use the funtion alternative_split_to_Jet_Num_Help to have the indices of the rows to put in each submatrix
    zero_indices, one_indices, two_indices, three_indices = split_to_Jet_Num_Help(tX)
    # for each group of indices, we construct the submatrix and we drop the column corresponding to the pri-jet number
    # we drop the column because it will not convey any useful information when we construct 4 different models
    tX_0 = tX[zero_indices, :]
    tX_0 = np.delete(tX_0, 22, axis=1)
    tX_1 = tX[one_indices, :]
    tX_1 = np.delete(tX_1, 22, axis=1)
    tX_2 = tX[two_indices, :]
    tX_2 = np.delete(tX_2, 22, axis=1)
    tX_3 = tX[three_indices, :]
    tX_3 = np.delete(tX_3, 22, axis=1)
    return tX_0, tX_1, tX_2, tX_3


def split_labels_to_Jet_Num(y, tX):
    """
    The function split the labels in four groups, each of them contains the label corresponding to the four
    submatrices created before
    :param y: the vector containing all the labels
    :param tX: the matrix (training or testing)
    :return: y_0,y_1,y_2,y_3 the four vectors of labels corresponding to the four submatrices
    """
    # we use the funtion alternative_split_to_Jet_Num_Help to have the indices of the rows to put in each submatrix
    zero_indices, one_indices, two_indices, three_indices = split_to_Jet_Num_Help(tX)
    # using these indices we construct the four vectors of labels
    y_0 = y[zero_indices]
    y_1 = y[one_indices]
    y_2 = y[two_indices]
    y_3 = y[three_indices]
    return y_0, y_1, y_2, y_3


def find_mass(tX):
    """
    Take the indices where the mass is not calculated, add the column which has 0 in those indices
    and 1 everywhere else
    :param tX: the input matrix (training or testing)
    :return: a new matrix tX which has a one more binary (0-1) column whose elements are 1 if the mass in the
    corresponding row is calculated (is different from -999.) 0 otherwise.
    """
    zero_indices = np.where(tX[:, 1] == -999.)[0]  # indices where the mass is Nan
    column_to_add = np.array([0 if i in zero_indices else 1 for i in range(tX.shape[0])])  # we create the column to add
    tX = np.insert(tX, 0, column_to_add, axis=1)  # we add the column
    return tX


def fix_array(tX, delete=None):
    """
    The function replaces the outliers with the mean of the column in case delete
    is not None, it erases columns that are invalid (all values = -999.)
    :param tX: the input matrix
    :param delete: if it is not none it erases columns that are invalid (all values = -999.)
    :return: tX, col_to_delete: the new matrix with the outliers treated opportunely and the columns deleted
    """
    if delete is not None:
        col_to_delete = []
    for i in range(1, tX.shape[1]):  # the first column will be valid since it is the column with the mass
        # check if the column has any valid indices
        index_column_valid = np.where(tX[:, i] != -999.)[0]
        if len(index_column_valid) == 0:
            # we drop the column (we will have to do the same for the test set as well)
            if delete is not None:
                col_to_delete.append(i)  # we save the column in the list
            else:
                print("You have an invalid column all values are -999.:", i)
        else:
            # for each column we find the 25% and 75% quartiles
            column_25_quantile, column_75_quantile = np.quantile(tX[index_column_valid, i],
                                                                 np.array([0.25, 0.75]))
            # for each column we find the interquartile
            interquantile = column_75_quantile - column_25_quantile
            # we find the 15% and 85% quartiles
            column_15_quantile, column_85_quantile = np.quantile(tX[index_column_valid, i],
                                                                 np.array([0.15, 0.85]))
            # if the values is >= 85% quartile + 1.5 interquartile or < 15% quartile - 1.5 interquartile it is
            # considered an outlier.
            indices_outliers = np.where((column_15_quantile - 1.5 * interquantile >= tX[index_column_valid, i])
                                        | (tX[index_column_valid, i] >=
                                           column_85_quantile + 1.5 * interquantile))[0]
            # we calculate then the median of the column
            median = np.median(tX[index_column_valid, i], axis=0)
            # we replace the outliers with the mean
            tX[index_column_valid[indices_outliers], i] = median
    if delete is not None:
        if delete == 0:
            col_to_delete.append(tX.shape[1] - 1)
        tX = np.delete(tX, col_to_delete, axis=1)  # we delete the columns with all Nan values
    return tX, col_to_delete


def fix_median(tX):
    """
    The function replaces the Nan values (-999.) with the median of the corresponding column
    :param tX: the training matrix
    :return: tX the matrix without any NaN values and a vector whose i-th component is the median corresponding to the
    (i+1)-th column
    """
    column_median = np.array([1])
    for i in range(1, tX.shape[1]):
        index_column_non_valid = np.where(tX[:, i] == -999.)[0]  # find invalid indices
        index_column_valid = np.where(tX[:, i] != -999.)[0]  # find valid indices
        median = np.median(tX[index_column_valid, i], axis=0)  # calculate the median of the valid indices
        column_median = np.hstack(
            [column_median, median])  # adding the median of the i-th column to substitute null values in the test
        tX[index_column_non_valid, i] = median  # replace all the invalid values with the median
    return tX, column_median


def fix_median_test(tX, median_values):
    """
    The function replaces the Nan values in the test set with the median computed in the training set for each column
    :param tX: test matrix
    :param median_values: median values of each (a part from the first one) column of the training set
    :return: tX the new test matrix
    """
    for i in range(1, tX.shape[1]):
        index_column_non_valid = np.where(tX[:, i] == -999.)[0]  # we find the indices with Nan values
        tX[index_column_non_valid, i] = median_values[i] # we replace them by the median
    return tX
