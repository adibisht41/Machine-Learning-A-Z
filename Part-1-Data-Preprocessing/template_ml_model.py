""" Template for Machine learning model """

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:, :-1].values #excluding the last column and including only independent variables(first 3 col)
y=dataset.iloc[:, 3].values

#splitting data : training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0);

#feature scalling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler();
X_train=sc_X.fit_transform(X_train);
X_test=sc_X.transform(X_test);


