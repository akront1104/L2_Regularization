
# coding: utf-8

# ## CISC 6930: Data Mining 
# Assignmnet 1
# <br>
# Angela Krontiris
# <br>
# Program used: Python
# <br>
# September 28, 2018

# In[256]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dataset 1 (100 rows and 10 columns)
df_train_1 = pd.read_csv('train-100-10.csv')
df_test_1 = pd.read_csv('test-100-10.csv')

# Dataset 2 (100 rows and 100 columns)
df_train_2 = pd.read_csv('train-100-100.csv')
df_test_2 = pd.read_csv('test-100-100.csv')

# Dataset 3 (1000 rows and 100 columns)
df_train_3 = pd.read_csv('train-1000-100.csv')
df_test_3 = pd.read_csv('test-1000-100.csv')

# Dataset 4-6 - Spit from Dataset 3 (1000 rowas and 100 columns)
df_train_3[0:50].to_csv('train-50(1000)-100.csv', index=False)
df_train_3[0:100].to_csv('train-100(1000)-100.csv', index=False)
df_train_3[0:150].to_csv('train-150(1000)-100.csv', index=False)

# Dataset 3 - Train Data Split
df_train_50 = pd.read_csv('train-50(1000)-100.csv')
df_train_100 = pd.read_csv('train-100(1000)-100.csv')
df_train_150 = pd.read_csv('train-150(1000)-100.csv')


# In[257]:

# df_train_1.head()


# In[258]:

# Convert DataFrames to Matrices
def dataframe_to_matrix(df):
    """Converts each dataframe to a matrix. Insert column of 1's. Create X_train, Y_train, X_test, Y_test"""
    #1) add column of ones to dataframe
    df.insert(0, '1', 1)
    
    # Extracting all features (x1,x2,...,xD)
    X_train = df.iloc[:,:-1]
    
    # Convert DataFrame to array
    X_matrix_train = X_train.values
    
    # Extract 'y' column
    Y_train = df.iloc[:,-1:]
    
    # Convert DataFrame to array
    Y_matrix_train = Y_train.values
    
    return(X_matrix_train, Y_matrix_train)


# In[259]:

# Set X and Y matrix to variables from each dataframe (Training Data)
x_matrix_train_1, y_matrix_train_1 = dataframe_to_matrix(df_train_1)
x_matrix_train_2, y_matrix_train_2 = dataframe_to_matrix(df_train_2)
x_matrix_train_3, y_matrix_train_3 = dataframe_to_matrix(df_train_3)

# Set X and Y matrix to variables from each dataframe (Testing Data)
x_matrix_test_1, y_matrix_test_1 = dataframe_to_matrix(df_test_1)
x_matrix_test_2, y_matrix_test_2 = dataframe_to_matrix(df_test_2)
x_matrix_test_3, y_matrix_test_3 = dataframe_to_matrix(df_test_3)

# Additional Datasets: Training Data Split from Dataset 3
x_matrix_train_50, y_matrix_train_50 = dataframe_to_matrix(df_train_50)
x_matrix_train_100, y_matrix_train_100 = dataframe_to_matrix(df_train_100)
x_matrix_train_150, y_matrix_train_150 = dataframe_to_matrix(df_train_150)


# In[260]:

def weight_calculation(X_train, Y_train, lambda_end, lambda_start = 0):
    """Calculates the weights of linear regression with lambda ranging from a to b"""
    
    # Specify lambda start and lambda end
    lambda_values = np.arange(lambda_start,lambda_end + 1)
    
    # List to store weights with varying lambdas
    weights_varying_lambda =[]

    for lambda_value in lambda_values:
        # Transposing X_train, give as a numpy array
        X_train_transpose = np.transpose(X_train) 
    
        #Multiply X'*X
        XTX_train =  np.dot(X_train_transpose, X_train)
    
        # Identity Matrix
        I = np.identity(len(XTX_train))

        # Multiply lambda by the identity Matrix
        Lambda_times_I = np.dot(lambda_value, I)
    
        # Add X'X Matrix and lambda*I Matrix
        XTX_plus_I = XTX_train +  Lambda_times_I
    
        # Take the multiplicative inverse of the matrix (X'X + lambda*I)
        XTX_plus_I_inv = np.linalg.inv(XTX_plus_I)
    
        # Mutliply X'*Y
        XTY_train = np.dot(X_train_transpose, Y_train)
    
        # Solve for w = (X'X + lambda(I))^-1 * X'Y
        weights = np.dot(XTX_plus_I_inv, XTY_train)
        
        # Flatten the array
        weights_varying_lambda.append(weights.flatten())
        
        # Weights for each lambda are stored in the array column wise
        weights_train_dataset = np.transpose(np.array(weights_varying_lambda))
        
    return(weights_train_dataset)


# ## Weight Calculations: 

# ### 1) Dataset 1:

# In[261]:

weights_train_dataset_1 = weight_calculation(x_matrix_train_1, y_matrix_train_1,lambda_end=150, lambda_start = 0)
# print(weights_train_dataset_1)


# ### 2) Dataset 2

# In[262]:

# Lambda range from 0 to 150
weights_train_dataset_2 = weight_calculation(x_matrix_train_2, y_matrix_train_2,lambda_end=150, lambda_start = 0)
# print(weights_train_dataset_2)


# In[263]:

# Lambda range from 1 to 150
weights_train_dataset_2_lambda_1_150 = weight_calculation(x_matrix_train_2, y_matrix_train_2,lambda_end=150, lambda_start = 1)
# print(weights_train_dataset_2_lambda_1_150)


# ### 3) Dataset 3

# In[264]:

# Lambda range from 0 to 150
weights_train_dataset_3 = weight_calculation(x_matrix_train_3, y_matrix_train_3,lambda_end=150, lambda_start = 0)
# print(weights_train_dataset_3)


# In[265]:

# Lambda range from 1 to 150
weights_train_dataset_3_lambda_1_150 = weight_calculation(x_matrix_train_3, y_matrix_train_3,lambda_end=150, lambda_start = 1)
# print(weights_train_dataset_3_lambda_1_150)


# ### Dataset 4
# Train-50(1000)-100

# In[266]:

# Lambda range from 0 to 150
weights_train_dataset_50_100 = weight_calculation(x_matrix_train_50, y_matrix_train_50,lambda_end=150, lambda_start = 0)
# print("Weights For Training Dataset 50(1000)-100:\n Lambda 0 to 150\n\n", weights_train_dataset_50_100)

# Lambda range from 1 to 150
weights_train_dataset_50_100_lambda_1_150 = weight_calculation(x_matrix_train_50, y_matrix_train_50,lambda_end=150, lambda_start = 1)
# print("\n\nWeights For Training Dataset 50(1000)-100:\n Lambda 1 to 150\n\n", weights_train_dataset_50_100_lambda_1_150)


# ### Dataset 5
# Train-100(1000)-100 

# In[267]:

# Lambda range from 0 to 150
weights_train_dataset_100_100 = weight_calculation(x_matrix_train_100, y_matrix_train_100,lambda_end=150, lambda_start = 0)
# print("Weights For Training Dataset 100(1000)-100:\n Lambda 0 to 150\n\n", weights_train_dataset_100_100)

# Lambda range from 1 to 150
weights_train_dataset_100_100_lambda_1_150 = weight_calculation(x_matrix_train_100, y_matrix_train_100,lambda_end=150, lambda_start = 1)
# print("\n\nWeights For Training Dataset 100(1000)-100:\n Lambda 1 to 150\n\n", weights_train_dataset_100_100_lambda_1_150)


# ### Dataset 6
# Train-150(1000)-100 

# In[268]:

# Lambda range from 0 to 150
weights_train_dataset_150_100 = weight_calculation(x_matrix_train_150, y_matrix_train_150,lambda_end=150, lambda_start = 0)
# print("Weights For Training Dataset 150(1000)-100:\n Lambda 0 to 150\n\n", weights_train_dataset_150_100)

# Lambda range from 1 to 150
weights_train_dataset_150_100_lambda_1_150 = weight_calculation(x_matrix_train_150, y_matrix_train_150,lambda_end=150, lambda_start = 1)
# print("\n\nWeights For Training Dataset 150(1000)-100:\n Lambda 1 to 150\n\n", weights_train_dataset_150_100_lambda_1_150)


# Function to calculate the mean squared error using the weights calculattion to make the predictions (y_hat). 

# In[269]:

def mean_squared_error(X_train, weights, Y_train):
    """Calculates the mean squared error (MSE) by taking the difference of 
    the predicted and actual values and squaring the values, then taking the average"""
    
    # Predictions (y^ = X*w)
    Y_pred = np.dot(X_train, weights)
    
    # Set mean squared error to zero
    sum_error = 0.0
    
    for i in range(len(Y_train)):
        Y_pred_error = Y_pred[i] - Y_train[i]
        sum_error += (Y_pred_error ** 2)
    MSE = sum_error / float(len(Y_train))
    
    return MSE


# ## Training and Test Mean Squared Error (MSE):

# ### Dataset 1

# In[270]:

# MSE of the training data 100-10, lambda ranging from 0 to 150
MSE_train_dataset_1 = mean_squared_error(x_matrix_train_1, weights_train_dataset_1, y_matrix_train_1)
print("MSE of Dataset, Train-100-10, lambda 0 to 150\n\n", MSE_train_dataset_1)

# MSE of the testing data 100-10, lambda ranging from 0 to 150
MSE_test_dataset_1 = mean_squared_error(x_matrix_test_1, weights_train_dataset_1, y_matrix_test_1)
print("MSE of Dataset, Test-100-10, lambda 0 to 150\n\n", MSE_test_dataset_1)


# ### Dataset 2

# In[271]:

# MSE of the training data 100-100, lambda ranging from 0 to 150
MSE_train_dataset_2 = mean_squared_error(x_matrix_train_2, weights_train_dataset_2, y_matrix_train_2)
print("MSE of Dataset, Train-100-100, lambda 0 to 150\n\n", MSE_train_dataset_2)

# MSE of the testing data 100-100, lambda ranging from 0 to 150
MSE_test_dataset_2 = mean_squared_error(x_matrix_test_2, weights_train_dataset_2, y_matrix_test_2)
print("MSE of Dataset, Test-100-100, lambda 0 to 150\n\n", MSE_test_dataset_2)


# In[272]:

# MSE of the training data 100-100, lambda ranging from 1 to 150
MSE_train_dataset_2_lambda_1_150 = mean_squared_error(x_matrix_train_2, weights_train_dataset_2_lambda_1_150, y_matrix_train_2)
print("MSE of Dataset, Train-100-100, lambda 1 to 150\n\n", MSE_train_dataset_2_lambda_1_150)

# MSE of the testing data 100-100, lambda ranging from 1 to 150
MSE_test_dataset_2_lambda_1_150 = mean_squared_error(x_matrix_test_2, weights_train_dataset_2_lambda_1_150, y_matrix_test_2)
print("MSE of Dataset, Test-100-100, lambda 1 to 150\n\n", MSE_test_dataset_2_lambda_1_150)


# ### Dataset 3

# In[273]:

# MSE of the training data 1000-100, lambda ranging from 0 to 150
MSE_train_dataset_3 = mean_squared_error(x_matrix_train_3, weights_train_dataset_3, y_matrix_train_3)
print("MSE of Dataset, Train-1000-100, lambda 0 to 150\n\n", MSE_train_dataset_3)

# MSE of the testing data 1000-100, lambda ranging from 0 to 150
MSE_test_dataset_3 = mean_squared_error(x_matrix_test_3, weights_train_dataset_3, y_matrix_test_3)
print("MSE of Dataset, Test-1000-100, lambda 0 to 150\n\n", MSE_test_dataset_3)


# In[274]:

# MSE of the testing data 1000-100, lambda ranging from 1 to 150
MSE_test_dataset_3_lambda_1_150 = mean_squared_error(x_matrix_test_3, weights_train_dataset_3_lambda_1_150, y_matrix_test_3)
print("MSE of Dataset, Test-1000-100, lambda 1 to 150\n\n", MSE_test_dataset_3_lambda_1_150)


# ### Dataset 4
# Train-50(1000)-100 

# In[275]:

# MSE of the training data 50(1000)-100, lambda ranging from 0 to 150
MSE_train_dataset_50_100 = mean_squared_error(x_matrix_train_50, weights_train_dataset_50_100, y_matrix_train_50)
print("MSE of Dataset, Train-50(1000)-100, lambda 0 to 150\n\n", MSE_train_dataset_50_100)

# MSE of the testing data (1000)-100, lambda ranging from 0 to 150
MSE_test_dataset_50_100 = mean_squared_error(x_matrix_test_3, weights_train_dataset_50_100, y_matrix_test_3)
print("MSE of Dataset, Test (1000)-100, lambda 0 to 150\n\n", MSE_test_dataset_50_100)

# MSE of the training data 50(1000)-100, lambda ranging from 1 to 150
MSE_train_dataset_50_100_lambda_1_150 = mean_squared_error(x_matrix_train_50, weights_train_dataset_50_100_lambda_1_150, y_matrix_train_50)
print("MSE of Dataset, Train-50(1000)-100, lambda 1 to 150\n\n", MSE_train_dataset_50_100_lambda_1_150)

# MSE of the testing data 50(1000)-100, lambda ranging from 1 to 150
MSE_test_dataset_50_100_lambda_1_150 = mean_squared_error(x_matrix_test_3, weights_train_dataset_50_100_lambda_1_150, y_matrix_test_3)
print("MSE of Dataset, Test (1000)-100, lambda 1 to 150\n\n", MSE_test_dataset_50_100_lambda_1_150)


# ### Dataset 5
# Train-100(1000)-100 

# In[276]:

# MSE of the training data 100(1000)-100, lambda ranging from 0 to 150
MSE_train_dataset_100_100 = mean_squared_error(x_matrix_train_100, weights_train_dataset_100_100, y_matrix_train_100)
print("MSE of Dataset, Train-100(1000)-100, lambda 0 to 150\n\n", MSE_train_dataset_100_100)

# MSE of the testing data (1000)-100, lambda ranging from 0 to 150
MSE_test_dataset_100_100 = mean_squared_error(x_matrix_test_3, weights_train_dataset_100_100, y_matrix_test_3)
print("MSE of Dataset, Test (1000)-100, lambda 0 to 150\n\n", MSE_test_dataset_100_100)

# MSE of the training data 100(1000)-100, lambda ranging from 1 to 150
MSE_train_dataset_100_100_lambda_1_150 = mean_squared_error(x_matrix_train_100, weights_train_dataset_100_100_lambda_1_150, y_matrix_train_100)
print("MSE of Dataset, Train-100(1000)-100, lambda 1 to 150\n\n", MSE_train_dataset_100_100_lambda_1_150)

# MSE of the testing data 100(1000)-100, lambda ranging from 1 to 150
MSE_test_dataset_100_100_lambda_1_150 = mean_squared_error(x_matrix_test_3, weights_train_dataset_100_100_lambda_1_150, y_matrix_test_3)
print("MSE of Dataset, Test (1000)-100, lambda 1 to 150\n\n", MSE_test_dataset_100_100_lambda_1_150)


# ### Dataset 6
# Train-150(1000)-100

# In[277]:

# MSE of the training data 150(1000)-100, lambda ranging from 0 to 150
MSE_train_dataset_150_100 = mean_squared_error(x_matrix_train_150, weights_train_dataset_150_100, y_matrix_train_150)
print("MSE of Dataset, Train-150(1000)-100, lambda 0 to 150\n\n", MSE_train_dataset_150_100)

# MSE of the testing data (1000)-100, lambda ranging from 0 to 150
MSE_test_dataset_150_100 = mean_squared_error(x_matrix_test_3, weights_train_dataset_150_100, y_matrix_test_3)
print("MSE of Dataset, Test (1000)-100, lambda 0 to 150\n\n", MSE_test_dataset_150_100)


# ## 1) MSE Plots:
# The plots generated below show the MSE on the training and testing data with lambdas ranging from 0 to 150 using six datasets.

# ### Dataset 1

# In[278]:

# Plot the mean squared error on the training and testing dataset 1
MSE_train_dataset_1_plot = plt.plot(MSE_train_dataset_1, label='Train MSE', color = 'blue')
MSE_test_dataset_1_plot = plt.plot(MSE_test_dataset_1, label='Test MSE', color = 'green')
plt.title("Dataset 1:\n Train-100-10  vs Test-100-10\n lambda= [0-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()


# ### Dataset 2

# In[279]:

# Plot the mean squared error on the training and testing dataset 2
MSE_train_dataset_2_plot = plt.plot(MSE_train_dataset_2, label='Train MSE', color = 'blue')
MSE_test_dataset_2_plot = plt.plot(MSE_test_dataset_2, label='Test MSE', color = 'green')
plt.title("Dataset 2:\n Train-100-100  vs Test-100-100\n lambda= [0-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()


# ### Dataset 3

# In[280]:

# Plot the mean squared error on the training and testing dataset 3
MSE_train_dataset_3_plot = plt.plot(MSE_train_dataset_3, label='Train MSE', color = 'blue')
MSE_test_dataset_3_plot = plt.plot(MSE_test_dataset_3, label='Test MSE', color = 'green')
plt.title("Dataset 3:\n Train-1000-100  vs Test-1000-100\n lambda= [0-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()


# ### Dataset 4

# In[281]:

# Plot the mean squared error on the train data 50(1000)-100 and test dataset 4 (1000-100)
MSE_train_dataset_50_100_plot = plt.plot(MSE_train_dataset_50_100, label='Train MSE', color = 'blue')
MSE_test_dataset_4_plot = plt.plot(MSE_test_dataset_50_100, label='Test MSE', color = 'green')
plt.title("Dataset 4:\n Train-50(1000)-100  vs Test-1000-100\n lambda= [0-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()


# ### Dataset 5

# In[282]:

# Plot the mean squared error on the train data 100(1000)-100 and test dataset 5 (1000-100)
MSE_train_dataset_100_100_plot = plt.plot(MSE_train_dataset_100_100, label='Train MSE', color = 'blue')
MSE_test_dataset_5_plot = plt.plot(MSE_test_dataset_100_100, label='Test MSE', color = 'green')
plt.title("Dataset 5:\n Train-100(1000)-100  vs Test-1000-100\n lambda= [0-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()


# ### Dataset 6

# In[283]:

# Plot the mean squared error on the train data 150(1000)-100 and test dataset 3 (1000-100)
MSE_train_dataset_150_100_plot = plt.plot(MSE_train_dataset_150_100, label='Train MSE', color = 'blue')
MSE_test_dataset_6_plot = plt.plot(MSE_test_dataset_150_100, label='Test MSE', color = 'green')
plt.title("Dataset 6:\n Train-150(1000)-100  vs Test-1000-100\n lambda= [0-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()


# ### 1a) For each dataset, which Lambda gives the least test set MSE?

# In[141]:

# # Find the mininum x (lambda) value for test dataset 1
# lambda_dataset_1 = MSE_test_dataset_1.argmin()

# print("For Test Dataset 1, lambda =", lambda_dataset_1, "gives the least MSE.")
# print(MSE_test_dataset_1[lambda_dataset_1])


# In[284]:

MSE_values = [MSE_test_dataset_1, MSE_test_dataset_2, MSE_test_dataset_3, MSE_test_dataset_50_100, MSE_test_dataset_100_100, MSE_test_dataset_150_100]

for MSE_value in MSE_values:
    lambda_min = MSE_value.argmin()
    MSE_min = MSE_value[lambda_min]
    
    print("lambda =", lambda_min, "gives the least MSE =", MSE_min)
    


# ### 1b) For datasets, train-100-100, train-50(1000)-100, and train-100(1000)-100, provide an additional graph with lambda ranging from 1 to 150

# In[285]:

# Plot the mean squared error on the train and test dataset 2 (100-100)
MSE_train_dataset_2_lambda_1_150_plot = plt.plot(MSE_train_dataset_2_lambda_1_150, label='Train MSE', color = 'blue')
MSE_test_dataset_2_lambda_1_150_plot = plt.plot(MSE_test_dataset_2_lambda_1_150, label='Test MSE', color = 'green')
plt.title("Train-100-100  vs Test-100-100\n lambda= [1-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Plot the mean squared error on the train set 50(1000)-100 and test set 4 (1000-100)
MSE_train_dataset_50_100_lambda_1_150_plot = plt.plot(MSE_train_dataset_50_100_lambda_1_150, label='Train MSE', color = 'blue')
MSE_test_dataset_4_lambda_1_150_plot = plt.plot(MSE_test_dataset_50_100_lambda_1_150, label='Test MSE', color = 'green')
plt.title("Train-50(1000)-100  vs Test-1000-100\n lambda= [1-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Plot the mean squared error on the train set 100(1000)-100 and test set 5 (1000-100)
MSE_train_dataset_100_100_lambda_1_150_plot = plt.plot(MSE_train_dataset_100_100_lambda_1_150, label='Train MSE', color = 'blue')
MSE_test_dataset_5_lambda_1_150_plot = plt.plot(MSE_test_dataset_100_100_lambda_1_150, label='Test MSE', color = 'green')
plt.title("Train-100(1000)-100  vs Test-1000-100\n lambda= [1-150]")
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()


# ### 1c) Explain why lambda=0 (i.e., no regularization) gives abnormally large MSEs for those three datasets in (b)

# Setting lambda to zero removes regularization completely. In this case, training focuses exclusively on minimizing loss, which poses the highest possible over fitting risk. 

# ### 2) Implement the 10-fold CV technique to select the best lambda value from the training set

# In[144]:

# print(x_matrix_train_1)


# In[145]:

# print(y_matrix_train_1)


# ### CV for Training Dataset 1: 100-10

# In[286]:

KFolds=10
# Find the fold size - Length of y matrix divided by number of folds
fold_size = int(len(y_matrix_train_1)/KFolds)

#MSE Sum 
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


# In[287]:

# print(MSE_test_1)


# In[288]:

# Best lambda for min test MSE value, 100-10, lambda:0 - 150
lambda_min_test_1 = MSE_test_1.argmin()
MSE_min_test_1 = MSE_test_1[lambda_min_test_1]

print("Dataset 1 (100-10): lambda =", lambda_min_test_1, "gives the least test MSE =", MSE_min_test_1)


# ### CV for Training Dataset 2: 100-100

# In[289]:

# Find the fold size - Length of y matrix divided by number of folds
fold_size_train_2 = int(len(y_matrix_train_2)/KFolds)

#MSE Sum 
MSE_sum_test_2 = 0 

#CV for Train-100-100, lambda: 0-150
for i in range(KFolds):
    X_test_2_fold = x_matrix_train_2[ i*fold_size_train_2 : (i+1)*fold_size_train_2]
    Y_test_2_fold = y_matrix_train_2[ i*fold_size_train_2 : (i+1)*fold_size_train_2]
    
    X_train_2_fold = np.concatenate(( x_matrix_train_2[ : i*fold_size_train_2], x_matrix_train_2[ (i+1)*fold_size_train_2 : ]), axis=0)
    Y_train_2_fold = np.concatenate(( y_matrix_train_2[ : i*fold_size_train_2], y_matrix_train_2[ (i+1)*fold_size_train_2 : ]), axis=0)
    
    weights_train_2 = weight_calculation(X_train_2_fold, Y_train_2_fold, 150, lambda_start = 0)
    
    MSE_sum_test_2 += mean_squared_error(X_test_2_fold, weights_train_2, Y_test_2_fold)

MSE_test_2 = MSE_sum_test_2/KFolds


# In[290]:

# print(MSE_train_2)


# In[291]:

# Best lambda for min test MSE value, 100-100, lambda:0 - 150
lambda_min_test_2 = MSE_test_2.argmin()
MSE_min_test_2 = MSE_test_2[lambda_min_test_2]

print("Dataset 2 (100-100): lambda =", lambda_min_test_2, "gives the least test MSE =", MSE_min_test_2)


# ### CV for Training Dataset 3: 1000-100

# In[292]:

# Find the fold size - Length of y matrix divided by number of folds
fold_size_train_3 = int(len(y_matrix_train_3)/KFolds)

#MSE Sum 
MSE_sum_test_3 = 0 

#CV for Train-1000-100, lambda: 0-150
for i in range(KFolds):
    X_test_3_fold = x_matrix_train_3[ i*fold_size_train_3 : (i+1)*fold_size_train_3]
    Y_test_3_fold = y_matrix_train_3[ i*fold_size_train_3 : (i+1)*fold_size_train_3]
    
    X_train_3_fold = np.concatenate(( x_matrix_train_3[ : i*fold_size_train_3], x_matrix_train_3[ (i+1)*fold_size_train_3 : ]), axis=0)
    Y_train_3_fold = np.concatenate(( y_matrix_train_3[ : i*fold_size_train_3], y_matrix_train_3[ (i+1)*fold_size_train_3 : ]), axis=0)
    
    weights_train_3 = weight_calculation(X_train_3_fold, Y_train_3_fold, 150, lambda_start = 0)
    
    MSE_sum_test_3 += mean_squared_error(X_test_3_fold, weights_train_3, Y_test_3_fold)

MSE_test_3 = MSE_sum_test_3/KFolds


# In[293]:

# print(MSE_test_3)


# In[294]:

# Best lambda for min test MSE value, 1000-100, lambda:0 - 150
lambda_min_test_3 = MSE_test_3.argmin()
MSE_min_test_3 = MSE_test_3[lambda_min_test_3]

print("Dataset 3 (1000-100): lambda =", lambda_min_test_3, "gives the least test MSE =", MSE_min_test_3)


# ### CV for training Dataset 4: 50(1000)-100

# In[295]:

# Find the fold size - Length of y matrix divided by number of folds
fold_size_train_50 = int(len(y_matrix_train_50)/KFolds)

#MSE Sum 
MSE_sum_test_50 = 0 

#CV for Train-50(1000-100), lambda: 0-150
for i in range(KFolds):
    X_test_50_fold = x_matrix_train_50[ i*fold_size_train_50 : (i+1)*fold_size_train_50]
    Y_test_50_fold = y_matrix_train_50[ i*fold_size_train_50 : (i+1)*fold_size_train_50]
    
    X_train_50_fold = np.concatenate(( x_matrix_train_50[ : i*fold_size_train_50], x_matrix_train_50[ (i+1)*fold_size_train_50 : ]), axis=0)
    Y_train_50_fold = np.concatenate(( y_matrix_train_50[ : i*fold_size_train_50], y_matrix_train_50[ (i+1)*fold_size_train_50 : ]), axis=0)
    
    weights_train_50 = weight_calculation(X_train_50_fold, Y_train_50_fold, 150, lambda_start = 0)
    
    MSE_sum_test_50 += mean_squared_error(X_test_50_fold, weights_train_50, Y_test_50_fold)

MSE_test_50 = MSE_sum_test_50/KFolds


# In[296]:

# print(MSE_test_50)


# In[297]:

# Best lambda for min test MSE value, 50(1000)-100, lambda:0 - 150
lambda_min_test_50 = MSE_test_50.argmin()
MSE_min_test_50 = MSE_test_50[lambda_min_test_50]

print("Dataset 4 (50(1000)-100): lambda =", lambda_min_test_50, "gives the least test MSE =", MSE_min_test_50)


# ### CV for training Dataset 5: 100(1000)-100

# In[298]:

# Find the fold size - Length of y matrix divided by number of folds
fold_size_train_100 = int(len(y_matrix_train_100)/KFolds)

#MSE Sum 
MSE_sum_test_100 = 0 

#CV for Train-100(1000-100), lambda: 0-150
for i in range(KFolds):
    X_test_100_fold = x_matrix_train_100[ i*fold_size_train_100 : (i+1)*fold_size_train_100]
    Y_test_100_fold = y_matrix_train_100[ i*fold_size_train_100 : (i+1)*fold_size_train_100]
    
    X_train_100_fold = np.concatenate(( x_matrix_train_100[ : i*fold_size_train_100], x_matrix_train_100[ (i+1)*fold_size_train_100 : ]), axis=0)
    Y_train_100_fold = np.concatenate(( y_matrix_train_100[ : i*fold_size_train_100], y_matrix_train_100[ (i+1)*fold_size_train_100 : ]), axis=0)
    
    weights_train_100 = weight_calculation(X_train_100_fold, Y_train_100_fold, 150, lambda_start = 0)
    
    MSE_sum_test_100 += mean_squared_error(X_test_100_fold, weights_train_100, Y_test_100_fold)

MSE_test_100 = MSE_sum_test_100/KFolds


# In[299]:

# print(MSE_test_100)


# In[300]:

# Best lambda for min test MSE value, 100(1000)-100, lambda:0 - 150
lambda_min_test_100 = MSE_test_100.argmin()
MSE_min_test_100 = MSE_test_100[lambda_min_test_100]

print("Dataset 5 (100(1000)-100): lambda =", lambda_min_test_100, "gives the least test MSE =", MSE_min_test_100)


# ### CV for training Dataset 6: 150(1000)-100

# In[301]:

# Find the fold size - Length of y matrix divided by number of folds
fold_size_train_150 = int(len(y_matrix_train_150)/KFolds)

#MSE Sum 
MSE_sum_test_150 = 0 

#CV for Train-150(1000-100), lambda: 0-150
for i in range(KFolds):
    X_test_150_fold = x_matrix_train_150[ i*fold_size_train_150 : (i+1)*fold_size_train_150]
    Y_test_150_fold = y_matrix_train_150[ i*fold_size_train_150 : (i+1)*fold_size_train_150]
    
    X_train_150_fold = np.concatenate(( x_matrix_train_150[ : i*fold_size_train_150], x_matrix_train_150[ (i+1)*fold_size_train_150 : ]), axis=0)
    Y_train_150_fold = np.concatenate(( y_matrix_train_150[ : i*fold_size_train_150], y_matrix_train_150[ (i+1)*fold_size_train_150 : ]), axis=0)
    
    weights_train_150 = weight_calculation(X_train_150_fold, Y_train_150_fold, 150, lambda_start = 0)
    
    MSE_sum_test_150 += mean_squared_error(X_test_150_fold, weights_train_150, Y_test_150_fold)

MSE_test_150 = MSE_sum_test_150/KFolds


# In[302]:

# print(MSE_test_150)


# In[303]:

# Best lambda for min test MSE value, 150(1000)-100, lambda:0 - 150
lambda_min_test_150 = MSE_test_150.argmin()
MSE_min_test_150 = MSE_test_150[lambda_min_test_150]

print("Dataset 6 (150(1000)-100): lambda =", lambda_min_test_150, "gives the least test MSE =", MSE_min_test_150)


# ### 3) Learning Curve for our algorithm using dataset 3 (1000-100) for lambda=1,25,150

# In[310]:

# Takes 5 random indices from the matrix
idx = np.random.choice(len(x_matrix_train_1), 5, replace=False)
print(idx)
print(len(x_matrix_train_1))
print(x_matrix_train_1[48])


# In[305]:

# Define function for learning curve
def learning_curve(x_matrix_train, y_matrix_train, x_matrix_test, y_matrix_test, rep, size):
    """This function will draw random subsets of increasing sizes and record the performance (MSE) on the corresponding test set when training on these subsets"""
    for lambda_value in [1,25,150]:
        size_list = range(10,1000,size) 
        MSE_array_test = np.zeros(len(size_list))
        MSE_array_train = np.zeros(len(size_list))
        for i in range(len(size_list)):
            rep_list_test = []
            rep_list_train = []
            
            for j in range(rep):
                idx = np.random.choice(len(x_matrix_train), size_list[i], replace=False)
            
                weights = weight_calculation(x_matrix_train[idx], y_matrix_train[idx],lambda_end=lambda_value, lambda_start = lambda_value)
    
                MSE_test = mean_squared_error(x_matrix_test, weights, y_matrix_test)
                
                MSE_train = mean_squared_error(x_matrix_train, weights, y_matrix_train)
        
                rep_list_test.append(MSE_test)
                rep_list_train.append(MSE_train)
            
            MSE_array_test[i] = np.average(rep_list_test)
            MSE_array_train[i] = np.average(rep_list_train)
        
        plt.plot(size_list, MSE_array_test, label='MSE_test')
        plt.plot(size_list, MSE_array_train, label='MSE_train')
        plt.xlabel('Training Set Size')
        plt.ylabel('MSE with lambda =' + str(lambda_value))
        plt.legend()
        plt.show()
#     return MSE_array_test
    


# In[306]:

learning_curve(x_matrix_train_3, y_matrix_train_3, x_matrix_test_3, y_matrix_test_3, 30, 10)


# In[311]:

# x_matrix_train_1[64]


# In[253]:

# x_matrix_train_1[idx]

