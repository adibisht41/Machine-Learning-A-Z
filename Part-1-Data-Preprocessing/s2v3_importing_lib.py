import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset=pd.read_csv('Data.csv')
#print(dataset)
X=dataset.iloc[:, :-1].values #excluding the last column and including only independent variables(first 3 col)
y=dataset.iloc[:, 3].values

# mising data in the dataset: so we take mean of the column values for it
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer=imputer.fit(X[:, 1:3])
X[:, 1:3]=imputer.transform(X[:, 1:3])
#print(X)

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder();
X[:,0]=labelencoder_X.fit_transform(X[:,0]);  #not permanant change
onehotencoder=OneHotEncoder(categorical_features=[0]);
X=onehotencoder.fit_transform(X).toarray();
labelencoder_y= LabelEncoder();
y=labelencoder_y.fit_transform(y);

#splitting data : training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0);

#feature scalling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler();
X_train=sc_X.fit_transform(X_train);
X_test=sc_X.transform(X_test);
#after scaling values of X_tarin and X_test becomes [-1,1]
#even if the model is not based on eucledian distance, we should scale it as it would converge fast
#as in case of Decision Tree






