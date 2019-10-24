#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Loading the dataset
dataset = pd.read_csv('slrdata.csv')
X=dataset.iloc[:,0].values
Y=dataset.iloc[:,-1].values
print(X,'\n',Y)

#Separating Training set and Testing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

print('X_train=',X_train)
print('X_test=',X_test)
print('Y_train=',Y_train)
print('Y_test=',Y_test)

#Fitting Simple Linear Model to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor=regressor.fit(X_train,Y_train)

#Predicting Testset results
Y_pred=regressor.predict(X_test)
print(Y_pred)

#Visualizing the Training set
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience(Training set)')
plt.show()