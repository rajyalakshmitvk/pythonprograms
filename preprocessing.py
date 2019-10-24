import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('data.csv')

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

print(X,'\n',Y)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy ='mean',axis=0)
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
print(X)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lencoder_X=LabelEncoder()
X[:,0]=lencoder_X.fit_transform(X[:,0])
print(X)

onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
print(X)

lencoder_Y=LabelEncoder()
Y=lencoder_Y.fit_transform(Y)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
print(X_train)
print(Y_train)

print(X_test)
print(Y_test)

from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

print(X_train)

print(X_test)