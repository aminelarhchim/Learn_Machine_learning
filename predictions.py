import csv 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import re


train = pd.read_csv('sales_train.csv')
test = pd.read_csv('test.csv')
print(train)


##### What is going on 
print(train.head())


