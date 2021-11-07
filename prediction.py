import csv 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

train = pd.read_csv('DATA/train.csv')
test = pd.read_csv('DATA/test.csv')

#print(test.shape)
#print(test.head(2))

train_ID= train['Id']
test_ID= test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

X=train[['MSSubClass','LotFrontage']]
Y=train['SalePrice']

regr=LinearRegression()
regr.fit(X,Y)
predict=regr.predict([[60,1000]])
print(predict)



#print(train.describe().transpose())
#print(train['OverallCond'])













