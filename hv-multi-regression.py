# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Read the file "Advertising.csv"
df = pd.read_csv('Advertising.csv')


df.head()

mse_list = []

cols = [['TV'], ['Radio'], ['Newspaper'], ['TV', 'Radio'], ['TV', 'Newspaper'],
        ['TV', 'Radio', 'Newspaper'], ['Radio', 'Newspaper']]

# Loop over all the predictor combinations 
for i in cols:

    # Set each of the predictors from the previous list as x
    x = df.loc[:, i]
    print(x)
    print('----------------------------------------')
    
    
    # Set the "Sales" column as the reponse variable
    y = df[['Sales']]

    # Split the data into train-test sets with 80% training data and 20% testing data. 
    # Set random_state as 0
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=0)

    # Initialize a Linear Regression model
    lreg = LinearRegression()

    # Fit the linear model on the train data
    lreg.fit(x_train, y_train)

    # Predict the response variable for the test set using the trained model
    y_pred = lreg.predict(x_test)
    
    # Compute the MSE for the test data
    MSE = mean_squared_error(y_test, y_pred)
    
    # Append the computed MSE to the initialized list
    mse_list.append(MSE)
    
    
# Helper code to display the MSE for each predictor combination
t = PrettyTable(['Predictors', 'MSE'])

for i in range(len(mse_list)):
    t.add_row([cols[i],round(mse_list[i],3)])

print(t)