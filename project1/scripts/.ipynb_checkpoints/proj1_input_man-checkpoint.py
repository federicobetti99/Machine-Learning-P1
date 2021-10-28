import numpy as np
from proj1_helpers import *

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
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

def create_output(tX_test,predictions_0, predictions_1, predictions_2_3):
    stacked_predictions = []
    count_0 = 0
    count_1 = 0
    count_2_3 = 0
    zero_indices,one_indices,two_three_indices = split_to_Jet_Num_Help(tX_test)
    for index_row in range(tX_test.shape[0]):
        if index_row in zero_indices:
            stacked_predictions.append(predictions_0[count_0])
            count_0 = count_0 + 1
        elif index_row in one_indices:
            stacked_predictions.append(predictions_1[count_1])
            count_1 = count_1 +1
        else:
            stacked_predictions.append(predictions_2_3[count_2_3])
            count_2_3 = count_2_3 + 1
    final_predictions = np.array([-1 if el < 0.5 else 1 for el in stacked_predictions])
    return final_predictions


# dividing the rows of tX by the number of jets, dropping the column Pri_Jet_Num and adding an extra column of np.ones
def split_to_Jet_Num_Help(tX):
    zero_indices = []
    one_indices = []
    two_three_indices = []
    zero_indices = np.where(tX[:,22]==0)[0]
    one_indices = np.where(tX[:,22]==1)[0]
    two_three_indices = np.where(np.logical_or(tX[:,22]==2, tX[:,22]==3))[0]
    return zero_indices,one_indices,two_three_indices

#Use indices to seperate the testing samples into the respective arrays
def split_to_Jet_Num(tX):
    zero_indices,one_indices,two_three_indices = split_to_Jet_Num_Help(tX)
    tX_0 = tX[zero_indices, :]
    tX_0 = np.delete(tX_0, 22, axis=1)
    tX_1 = tX[one_indices, :]
    tX_1 = np.delete(tX_1, 22, axis=1)
    tX_2_3 = tX[two_three_indices, :]
    return tX_0,tX_1,tX_2_3

#Split the labels according to the indices of the number of jets
def split_labels_to_Jet_Num(y,tX):
    zero_indices,one_indices,two_three_indices = split_to_Jet_Num_Help(tX)
    y_0 = y[zero_indices]
    y_1 = y[one_indices]
    y_2_3 = y[two_three_indices]
    return y_0,y_1,y_2_3

# take the indices where the mass is not calculated, add the column which has 0 in those indices
# and 1 everywhere else
def find_mass(tX):
    zero_indices = np.where(tX[:,1] == -999.)[0]
    column_to_add = np.array([0 if i in zero_indices else 1 for i in range(tX.shape[0])])
    tX = np.insert(tX, 0, column_to_add, axis=1)
    return tX

# Replacing the outliers in valid columns
# In case delete = 0 erase columns that are invalid (all values = -999.)
# initialize the list of columns that need to be deleted
def fix_array(tX,delete = None):
    if delete is not None:
        col_to_delete = []
    for i in range(1, tX.shape[1]):
        #check if the column has any valid indices
        index_column_valid =np.where(tX[:,i] != -999.)[0]
        if len(index_column_valid)==0:
            #we drop the column (we will have to do the same for the test set as well)
            if delete is not None:
                col_to_delete.append(i)
            else:
                print("You have an invalid column all values are -999.:",i)
        else :
            column_25_quantile, column_75_quantile = np.quantile(tX[index_column_valid,i],
                                                         np.array([0.25, 0.75]))
            interquantile = column_75_quantile-column_25_quantile
            column_15_quantile, column_85_quantile = np.quantile(tX[index_column_valid,i],
                                                         np.array([0.15, 0.85]))
            indices_outliers = np.where((column_15_quantile - 1.5 * interquantile >= tX[index_column_valid,i])
                                             | (tX[index_column_valid,i] >=
                                                column_85_quantile + 1.5 * interquantile))[0]
            median = np.median(tX[index_column_valid, i], axis = 0)
            tX[index_column_valid[indices_outliers],i] =  median
    if delete is not None:
        if delete == 0:
            col_to_delete.append(tX.shape[1]-1)
        tX = np.delete(tX, col_to_delete, axis=1)
        print(col_to_delete)
    print(tX.shape)
    return tX

#replace invalid values in the columns with the mean of the valid ones
def fix_mean(tX):
    for i in range(1, tX.shape[1]):
        index_column_non_valid =np.where(tX[:,i] == -999.)[0] #find invalid indices
        index_column_valid =np.where(tX[:,i] != -999.)[0] #find valid indices
        median = np.median(tX[index_column_valid, i], axis = 0) #calculate the median of the valid indices
        tX[index_column_non_valid,i] =  median #replace all the invalid values with the median
    return tX
