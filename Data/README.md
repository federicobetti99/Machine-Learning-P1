# Data Files
This folder should always contain:<br>

## data.zip
The zip with the **test.csv** and the **train.csv**<br>

### train.csv
The data used to train all the models

### test.csv
The test data with which we produce the test output, from each model.<br><br>

## Missing Files
If any of the afforementioned files is missing then you should create them using the following process, otherwise you will be unable to execute any of the code files.The process corresponding to each file is the following:<br>
1.git pull the **data.zip** if it is not present.<br>
2.extract everything from the **data.zip** in order to get the **train.csv** and the **test.csv**<br><br>

## Optional Files
This folder can also contain the following data files:<br>
(This text is meant to be used as a dictionary for all the different possible outputs,from each model)<br>

### submission_GD.csv
This file contains the predictions from the Gradient Descent Method on the Linear Model.

### submission_SGD.csv
This file contains the predictions from the Stochastic Gradient Descent Method on the Linear Model.

### submission_LS.csv
This file contains the predictions from the Least Squares regression using normal equations.

### submission_ridge.csv
This file contains the predictions from the Ridge Regression using normal equations.

### submission_logistic.csv
This file contains the predictions from the Logistic Regression Model using the Gradient Descent Method.

### submission_reg_logistic.csv
This file contains the predictions from the Regularized Logistic Regression using the Gradient Descent Method.

### submission_best.csv
This file contains the predictions from our best model the ensembled classifier and is the output of run.py
