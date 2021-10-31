# Files
In this directory we have the following files:
## implementations.py

This file includes the following functions:<br>

#### least_squares_GD(y, tx, initial_w, max_iters, gamma)
The implementation for linear regression using the gradient descent method with the mean squared error as an optimization function.

#### least_squares_SGD(y, tx, initial_w, max_iters, gamma)
The implementation for linear regression using the stochastic gradient descent method with mean squared error as the optimization function.

#### least_squares(y, tx)
The implementation for linear regression using normal equations derived by using the mean squared error, as an optimization function.

#### ridge_regression(y, tx, lambda )
The implementation for ridge regression.

#### logistic_regression(y, tx, initial_w, max_iters, gamma)
The implementation for logistic regression using the gradient descent method.

#### reg_logistic_regression(y, tx, lambda ,initial_w, max_iters, gamma)
The implementation for regularized logistic regression using the gradient descent method

## run.py
This script creates the csv which we used to submit to aircrowd and get an accuracy of 0.817, using our best model.<br>
The model is an ensemble of two ridge regressions and a logistic one.We train these models seperately.<br>
Then we parse the test samples with each model seperately and we utilize a hard voting scheme in order to combine the predictions in favor of the majority.

## Helper Files
The Code directory also contains several helper files which hold functions that are essential for our project. Namely We have the following files:

#### proj1_helpers.py
This file contains essential helper functions,required by numerous files:<br><br>
1.**ensemble_predictions** used for the hard voting scheme of the ensemble model<br>
2.**predict_labels**<br>
3.**compute_loss** calculates the MSE loss for the linear model<br>
4.**compute_gradient** calculates the gradient for the linear model with MSE as an objective function<br>
5.**compute_stoch_gradient** calculates the stochastic gradient for the linear model with MSE as an objective function<br>
6.**sigmoid** The implementation of the sigmoid function<br>
7.**calculate_loss** calculates loss for the Logistic Regression Model<br>
8.**calculate_gradient** calculates the gradient for the logistic regression model<br>
9.**compute_accuracy** computes the accuracy of a model given its predictions and the ground truth labels<br>
10.**build_poly** augments a feature vector by adding all the powers of every feature up to a given degree.<br>
11.**build_poly_cov_help** computes the cross feature expansion of degree 2, of a certain sample.<br>
12.**build_poly_cov** computes the cross feature expansion of a certain feature vector.<br>
13.**random_interval** calculates a number of random values within the specified interval.<br>
14.**build_k_indices** build the shuffled subvectors required for cross validation

#### proj1_input_manipulation.py
This file contains essential helper functions, required for input manipulation and preproseccing:<br><br>
1.**load_csv_data** Loads a csv<br>
2.**create_csv_submission** Creates a csv file for submission to aircrowd.<br>
3.**standardize** Standardize the training dataset and extract its mean value and the standard variance.<br>
4.**standardize_test** Standardize the test set using predefined means and variances.<br>
5.**create_output** Combine the 4 outputs for jets 0,1,2,3 into 1 output given the original order of the observations.<br>
6.**alternative_split_to_Jet_Num_Help** Find the indices for each different category of number of jets<br>
7.**alternative_split_to_Jet_Num** Split the feature vector into 4 vectors, one corresponding to each number of jets.<br>
8.**alternative_split_labels_to_Jet_Num** Split the labels vector into three subvectors one for each number of Jets.<br>
9.**find_mass** Seperate the indices with uncalculated mass with an additional column of 0s and 1s.<br>
10.**fix_array** Replaces the outliers of each columns and can delete invalid columns.<br>
11.**fix_median** This function replaces invalid values with the corresponding mean.<br>
12.**fix_median_test** This function replaces invalid values in a test set with the corresponding given mean.

#### proj1_linear_model.py
This file contains functions essential for the hyperparameter fitting of our models for linear regression:<br><br>
1.**cross_validation_GD** This function conducts cross validation for a given set of hyperparameters for the linear model, for the gradient descent method.<br>
2.**finetune_GD** This function takes a hyperparameter vector for the degree and finds the best hyperparameter out of the given ones<br>
3.**optimal_weights_GD** This function given all the hyperparameters and the training samples, computes the optimal weights, for this method.<br>
4.**predict_GD** This function given some weights and an augmentation degree, makes a prediction for the linear model.<br>
5.**cross_validation_SGD** This function conducts cross validation for a given set of hyperparameters for the linear model, for the stochastic gradient descent method.<br>
6.**finetune_SGD** This function takes a hyperparameter vector for the degree and finds the best hyperparameter out of the given ones<br>
7.**optimal_weights_SGD** This function given all the hyperparameters and the training samples, computes the optimal weights, for this method.<br>
8.**predict_SGD** This function given some weights and an augmentation degree, makes a prediction for the linear model.

#### proj1_ridge_regress.py
This file contains functions essential for the hyperparameter fitting of our models for ridge regression:<br><br>
1.**cross_validation_ridge** This function conducts cross validation for a given set of hyperparameters for ridge regression.<br>
2.**finetune_ridge** This function takes a hyperparameter vector for the degree and a vector for the lambda and finds the best hyperparameters out of the given ones<br>
3.**optimal_weights_ridge** This function given all the hyperparameters and the training samples, computes the optimal weights, for this method.<br>
4.**predict_ridge** This function given some weights and an augmentation degree, makes a prediction for the linear model.

#### proj1_logistic.py
This file contains functions essential for the hyperparameter fitting of our models for logistic regression:<br><br>
1.**cross_validation_logistic** This function conducts cross validation for a given set of hyperparameters for logistic regression.<br>
2.**finetune_logistic** This function takes a hyperparameter vector for the degree and a vector for the lambda and finds the best hyperparameters out of the given ones.<br>
3.**optimal_weights_logistic** This function given all the hyperparameters and the training samples, computes the optimal weights, for this method.<br>
4.**predict_logistic**This function given some weights and an augmentation degree, makes a prediction for the logistic model.

## Notebook Executions
The code directory also contains seperate notebooks, one for each function we were required to implement.By executing the corresponding notebook you can see the hyperparameter fitting for that process and you will also create the corresponding output file in the **Data** folder.Namely we have the following files:
1.**run_grad.ipynb** Executes a notebook that fits the parameters with the Gradient Descent Method for the linear model and makes the relative predictions.<br>
2.**run_stochastic_grad.ipynb** Executes a notebook that fits the parameters with the Stochastic Gradient Descent Method for the linear model and makes the relative predictions.<br>
3.**run_least_squares.ipynb** Executes a notebook that fits the parameters with the Least Squares Method for the linear model and makes the relative predictions.<br>
4.**run_ridge.ipynb** Executes a notebook that fits the parameters with the Ridge Regression for the linear model and makes the relative predictions.<br>
5.**run_logistic.ipynb** Executes a notebook that fits the parameters for Logistic Regression with the Gradient Descent for the logistic model and makes the relative predictions.<br>
6.**run_regularized_logistic.ipynb** Executes a notebook that fits the parameters for Regularized Logistic Regression with the Gradient Descent Method for the logistic model and makes the relative predictions.<br>
