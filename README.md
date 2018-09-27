# L2_Regularization

Question 1: Implement L2 regularized linear regression algorithm with Lambda ranging from 0 to 150 (integers only). For each of the six datasets, plot the training set MSE and test set MSE as a function of lambda (x-axis) in one graph. 

### First function defined - Dataframe to Matrix:
def dataframe_to_matrix(df):

    """Converts each dataframe to a matrix. Insert column of 1's. Create X_train, Y_train, X_test, Y_test"""
    
### Second function defined - Weight Calculation:
def weight_calculation(X_train, Y_train, lambda_end, lambda_start = 0):

    """Calculates the weights of linear regression with lambda ranging from a to b"""

### Third function defined - MSE Calucation:
def mean_squared_error(X_train, weights, Y_train):

    """Calculates the mean squared error (MSE) by taking the difference of 
    the predicted and actual values and squaring the values, then taking the average"""
    
### Choose lambda that gives the least test set MSE
    lambda_min = MSE_value.argmin()
    MSE_min = MSE_value[lambda_min]

### Plot the train and test MSE for lambda ranging from 0 to 150 (repeated for each dataet)
MSE_train_dataset_1_plot = plt.plot(MSE_train_dataset_1, label='Train MSE', color = 'blue')
MSE_test_dataset_1_plot = plt.plot(MSE_test_dataset_1, label='Test MSE', color = 'green')
plt.title("Dataset 1:\n Train-100-10  vs Test-100-10\n lambda= [0-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()


Question 2: Implement 10-fold CV technique to select the best lambda value from the training set

### Using CV with fold size to choose the best lambda and corresponding test MSE (run for each dataset)

fold_size = int(len(y_matrix_train_1)/KFolds)
KFolds=10
MSE_sum_test_1 = 0 

#CV for Train-100-10, lambda: 0-150
for i in range(KFolds):

    X_test_fold = x_matrix_train_1[ i*fold_size : (i+1)*fold_size]
    Y_test_fold = y_matrix_train_1[ i*fold_size : (i+1)*fold_size]
    
    X_train_fold = np.concatenate(( x_matrix_train_1[ : i*fold_size], x_matrix_train_1[ (i+1)*fold_size : ]), axis=0)
    Y_train_fold = np.concatenate(( y_matrix_train_1[ : i*fold_size], y_matrix_train_1[ (i+1)*fold_size : ]), axis=0)
    
    weights = weight_calculation(X_train_fold, Y_train_fold, 150, lambda_start = 0)
    MSE_sum_test_1 += mean_squared_error(X_test_fold, weights, Y_test_fold)

MSE_test_1 = MSE_sum_test_1/KFolds

Question 3: Fix lambda = 1, 25, 150. For each of these values, plot a learning curve for the algorithm using the dataset 1000-100

### Function to plot learning curve for each lambda
def learning_curve(x_matrix_train, y_matrix_train, x_matrix_test, y_matrix_test, rep, size):

    """This function will draw random subsets of increasing sizes and record the performance (MSE) 
        on the corresponding test set when training on these subsets"""
   
