
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset
dataset=pd.read_csv('data.csv')
print(dataset)
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values


#Missing Data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
print(X)
print(Y)

#Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
print(X)

labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)
print(Y)

onehotencoder_X=OneHotEncoder(categorical_features=[0])
X=onehotencoder_X.fit_transform(X).toarray()
print(X)
print('\n\n')

#Splitting training sets and testing sets
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2,random_state=0)
print(Xtrain)
print('\n\n')
print(Xtest)
print('\n\n')
print(Ytrain)
print('\n\n')
print(Ytest)

