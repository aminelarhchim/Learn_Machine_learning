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


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

y_train=train['Survived']

##### What is going on 

#print(train.info())    #type of data
#print(train.describe())  # mean, max, min..
#print(train.isnull().sum()) # missing values



# concatenate train and test
all=pd.concat([train,test],sort=False)


#Replacing the missing values with median 
all['Age'] = all['Age'].fillna(value=all['Age'].median())
all['Fare'] = all['Fare'].fillna(value=all['Fare'].median())

#Replacing the missing values with most common values
all['Embarked']=all['Embarked'].fillna('S')

# One-hot-encoding Age
all.loc[ all['Age'] <= 16, 'Age'] = 0
all.loc[(all['Age'] > 16) & (all['Age'] <= 32), 'Age'] = 1
all.loc[(all['Age'] > 32) & (all['Age'] <= 48), 'Age'] = 2
all.loc[(all['Age'] > 48) & (all['Age'] <= 64), 'Age'] = 3
all.loc[(all['Age'] > 64)  , 'Age'] = 4



def get_title(name):
    title_search = re.search(' ([A-Za-z]+\.)', name)
    
    if title_search:
        return title_search.group(1)
    return ""

all['Title'] = all['Name'].apply(get_title)


# Encoding Title
all['Title'] = all['Title'].replace(['Capt.', 'Dr.', 'Major.', 'Rev.','Col.'], 'Officer.')
all['Title'] = all['Title'].replace(['Lady.', 'Countess.', 'Don.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Royal.')
all['Title'] = all['Title'].replace(['Mlle.', 'Ms.'], 'Miss.')
all['Title'] = all['Title'].replace(['Mme.'], 'Mrs.')



#Family Size & Alone 
all['Family_Size'] = all['SibSp'] + all['Parch'] + 1
all['IsAlone'] = 0
all.loc[all['Family_Size']==1, 'IsAlone'] = 1


#Dropping unwanted variables
all= all.drop(['Name','Ticket'],axis=1)

#Get dummies
all= pd.get_dummies(all)


training_set= all[all['Survived'].notna()]
test_set= all[all['Survived'].isna()]
Id= test_set['PassengerId']
test_set=test_set.drop(['Survived','PassengerId'],axis=1)



#Train Split
X_train, X_test, y_train, y_test = train_test_split(training_set.drop(['PassengerId','Survived'],axis=1), 
                                                    training_set['Survived'], test_size=0.30, 
                                                    random_state=101)




# Building the random forest 
RF_Model = RandomForestClassifier()  
RF_Model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=7, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=6,
                       min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1,
                       oob_score=False, random_state=None, verbose=0,
                       warm_start=False)

# Fitting the model
RF_Model.fit(X_train, y_train)


#Gives the score for the training_set and the test_set

print(f'Test : {RF_Model.score(X_test, y_test):.3f}')
print(f'Train : {RF_Model.score(X_train, y_train):.3f}')

#making predictions
predictions = RF_Model.predict(test_set).astype(int)

# Give the predictions the right form and turn them into a csv file
prediction_Identified = pd.DataFrame({'PassengerId': Id, 'Survived': predictions })
prediction_Identified.to_csv("prediction.csv", index = False)

















