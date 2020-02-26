# import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# import dataset
dataset = pd.read_csv('sas.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
onehotencoder = OneHotEncoder(categories = 'auto')
y = onehotencoder.fit_transform(y.reshape(-1,1)).toarray()

# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# fitting multible linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict the test set result
y_pred = regressor.predict(X_test)

# find biggest num
max2 = []
for i in range(len(y_pred)):
  max1=0
  for j in range(len(y_pred[0])):
    if y_pred[i][j] > max1:
      max1 = y_pred[i][j]
  
  max2.append(max1)
  
# modify on prediction
for i in range(len(y_pred)):
  for j in range(len(y_pred[0])):
    if y_pred[i][j] >= max2[i]:
      y_pred[i][j] = 1
    else:
      y_pred[i][j] = 0
  
# return values
y_return1 = onehotencoder.inverse_transform(y_pred)      
y_return_final = labelencoder_y.inverse_transform(y_return1)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))