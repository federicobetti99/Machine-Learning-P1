# Importing the libraries
import numpy as np
import math
import random

# Our libraries
from proj1_helpers import *
from proj1_input_manipulation import *
from proj1_linear_model import *
from proj1_ridge_regress import *
from proj1_logistic import *

# Loading the Training Data in the feature matrix,class labels and event ids
DATA_TRAIN_PATH = '../Data/train.csv' # train data path here
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# changing the labels {-1,1} -> {0,1}
y_ = np.array([0 if l == -1 else 1 for l in y])

# dividing the features by the number of Jets
tX_0, tX_1, tX_2, tX_3 = split_to_Jet_Num(tX)

# dividing also the output by the number of Jets
y_0, y_1, y_2, y_3 = split_labels_to_Jet_Num(y_, tX)

# take the indices where the mass is not calculated, add the column which has 0 in those indices
# and 1 everywhere else for all submatrices 0,1,2_3
tX_0 = find_mass(tX_0)
tX_1 = find_mass(tX_1)
tX_2 = find_mass(tX_2)
tX_3 = find_mass(tX_3)

# dropping the entire nan columns if there are any
# for each column replace the outliers with its median
tX_0, col_to_delete_0 = fix_array(tX_0, 0)
tX_1, col_to_delete_1 = fix_array(tX_1, 1)
tX_2, _ = fix_array(tX_2, 2)
tX_3, _ = fix_array(tX_3, 3)

# substitute the -999 values with the median
tX_0, column_median_0 = fix_median(tX_0)
tX_1, column_median_1 = fix_median(tX_1)
tX_2, column_median_2 = fix_median(tX_2)
tX_3, column_median_3 = fix_median(tX_3)

# standardize the data, apart from the column of the mass added before
tX_3[:,1:], mean_3, std_3 = standardize(tX_3[:,1:])
tX_2[:,1:], mean_2, std_2 = standardize(tX_2[:,1:])
tX_0[:,1:], mean_0, std_0 = standardize(tX_0[:,1:])
tX_1[:,1:], mean_1, std_1 = standardize(tX_1[:,1:])

# insert the column for the bias term
tX_tilda_0 = np.insert(tX_0, 0, np.ones(tX_0.shape[0]), axis=1)
tX_tilda_1 = np.insert(tX_1, 0, np.ones(tX_1.shape[0]), axis=1)
tX_tilda_2 = np.insert(tX_2, 0, np.ones(tX_2.shape[0]), axis=1)
tX_tilda_3 = np.insert(tX_3, 0, np.ones(tX_3.shape[0]), axis=1)

# compute the weights for the 3 models of the ensemble
# using the precomputed optimal hyperparameters, by cross_validation

# model 1 ridge
w_ridge_0_model_1 = optimal_weights_ridge(tX_tilda_0, y_0, degree=2, lambda_=1e-05, crossing =True)
w_ridge_1_model_1 = optimal_weights_ridge(tX_tilda_1, y_1, degree=2, lambda_=0.00464159, crossing=True)
w_ridge_2_model_1 = optimal_weights_ridge(tX_tilda_2, y_2, degree=2, lambda_=0.00464159, crossing =True)
w_ridge_3_model_1 = optimal_weights_ridge(tX_tilda_3, y_3, degree=2, lambda_=2.78255940e-05, crossing=True)

#model 2 ridge
w_ridge_0_model_2 = optimal_weights_ridge(tX_tilda_0, y_0, degree=2, lambda_=5.58459201e-06, crossing =True)
w_ridge_1_model_2 = optimal_weights_ridge(tX_tilda_1, y_1, degree=2, lambda_=3.55553621e-05, crossing=True)
w_ridge_2_model_2 = optimal_weights_ridge(tX_tilda_2, y_2, degree=2, lambda_=7.24495921e-05, crossing =True)
w_ridge_3_model_2 = optimal_weights_ridge(tX_tilda_3, y_3, degree=2, lambda_=3.97155601e-05, crossing=True)

#the gamma for the logistic
gamma = 0.00017783
#model 3 regularized logistic
w_logistic_0_model_3 = optimal_weights_logistic(tX_tilda_0, y_0, gamma, degree=2, lambda_ = 7.6128834,crossing= True)
w_logistic_1_model_3 = optimal_weights_logistic(tX_tilda_1, y_1, gamma, degree=2, lambda_ = 9.38146586,crossing= True)
w_logistic_2_model_3 = optimal_weights_logistic(tX_tilda_2, y_2, gamma, degree=2, lambda_ = 6.44744181,crossing= True)
w_logistic_3_model_3 = optimal_weights_logistic(tX_tilda_3, y_3, gamma, degree=2, lambda_ = 5.08512828,crossing= True)

# open the test file
DATA_TEST_PATH = '../Data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#split the test feature matrix into the 4 subgroups
tX_test_0, tX_test_1, tX_test_2, tX_test_3 = split_to_Jet_Num(tX_test)

#adding a column of zeros and ones to detect whether the mass was measured or not
tX_test_0 = find_mass(tX_test_0)
tX_test_1 = find_mass(tX_test_1)
tX_test_2 = find_mass(tX_test_2)
tX_test_3 = find_mass(tX_test_3)

#drop the same columns, that we drop in the train set
tX_test_0 = np.delete(tX_test_0, col_to_delete_0, axis=1)
tX_test_1 = np.delete(tX_test_1, col_to_delete_1, axis=1)

#Substitute the -999 values of the remaining columns with the median of the train set
tX_test_0 = fix_median_test(tX_test_0, column_median_0)
tX_test_1 = fix_median_test(tX_test_1, column_median_1)
tX_test_2 = fix_median_test(tX_test_2, column_median_2)
tX_test_3 = fix_median_test(tX_test_3, column_median_3)

# we standardize everything a part from the column added manually
# we use the mean and the standard variance of the train set
tX_test_0[:,1:] = standardize_test(tX_test_0[:,1:], mean_0, std_0)
tX_test_1[:,1:] = standardize_test(tX_test_1[:,1:], mean_1, std_1)
tX_test_2[:,1:] = standardize_test(tX_test_2[:,1:], mean_2, std_2)
tX_test_3[:,1:] = standardize_test(tX_test_3[:,1:], mean_3, std_3)

#We insert the column for the bias term, all ones
tX_tilda_test_0 = np.insert(tX_test_0, 0, np.ones(tX_test_0.shape[0]), axis=1)
tX_tilda_test_1 = np.insert(tX_test_1, 0, np.ones(tX_test_1.shape[0]), axis=1)
tX_tilda_test_2 = np.insert(tX_test_2, 0, np.ones(tX_test_2.shape[0]), axis=1)
tX_tilda_test_3 = np.insert(tX_test_3, 0, np.ones(tX_test_3.shape[0]), axis=1)

#we generate the predictions for the 3 models of the ensemble

#model 1 ridge
predictions_ridge_0_model_1 = predict_ridge(tX_tilda_test_0, w_ridge_0_model_1, 2, True)
predictions_ridge_1_model_1 = predict_ridge(tX_tilda_test_1, w_ridge_1_model_1, 2, True)
predictions_ridge_2_model_1 = predict_ridge(tX_tilda_test_2, w_ridge_2_model_1, 2, True)
predictions_ridge_3_model_1 = predict_ridge(tX_tilda_test_3, w_ridge_3_model_1, 2, True)

#model 2 ridge
predictions_ridge_0_model_2 = predict_ridge(tX_tilda_test_0, w_ridge_0_model_2, 2, True)
predictions_ridge_1_model_2 = predict_ridge(tX_tilda_test_1, w_ridge_1_model_2, 2, True)
predictions_ridge_2_model_2 = predict_ridge(tX_tilda_test_2, w_ridge_2_model_2, 2, True)
predictions_ridge_3_model_2 = predict_ridge(tX_tilda_test_3, w_ridge_3_model_2, 2, True)

#model 3 ridge
predictions_logistic_0_model_3 = predict_logistic(tX_tilda_test_0, w_logistic_0_model_3, 2, True)
predictions_logistic_1_model_3 = predict_logistic(tX_tilda_test_1, w_logistic_1_model_3, 2, True)
predictions_logistic_2_model_3 = predict_logistic(tX_tilda_test_2, w_logistic_2_model_3, 2, True)
predictions_logistic_3_model_3 = predict_logistic(tX_tilda_test_3, w_logistic_3_model_3, 2, True)

#We generate the ensemble predictions using a hard voting scheme
average_predictions_0 = ensemble_predictions(predictions_ridge_0_model_1, predictions_ridge_0_model_2, predictions_logistic_0_model_3)
average_predictions_1 = ensemble_predictions(predictions_ridge_1_model_1, predictions_ridge_1_model_2, predictions_logistic_1_model_3)
average_predictions_2 = ensemble_predictions(predictions_ridge_2_model_1, predictions_ridge_2_model_2, predictions_logistic_2_model_3)
average_predictions_3 = ensemble_predictions(predictions_ridge_3_model_1, predictions_ridge_3_model_2, predictions_logistic_3_model_3)

#we merge the predictions
final_mixed_predictions = create_output(tX_test, average_predictions_0, average_predictions_1, average_predictions_2, average_predictions_3)

# name towards the output file
OUTPUT_PATH_ENSEMBLE= '../Data/submission_best.csv'

# print csv file according to ensemble results
create_csv_submission(ids_test, final_mixed_predictions, OUTPUT_PATH_ENSEMBLE)
