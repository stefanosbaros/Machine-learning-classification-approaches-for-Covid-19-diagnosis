#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 08:04:06 2021

@author: stefanosbaros
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,mean_squared_error, accuracy_score
from sklearn import metrics
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pickle


# In this script we perform feature engineering and model selection via l1 regularization
# First, we design new features by considering interactions among the main features and then carry out
# l1 regularization in order to decide which ones to keep


# loading data

main_file="/Users/stefanosbaros/Desktop/Covid_ML_project/"


Covid_path = '/Users/stefanosbaros/Desktop/Covid_ML_project/corona_tested_individuals_Israel.csv'
Covid_data = pd.read_csv(Covid_path)


# data pre-processing
Covid_data['corona_result']=np.where(Covid_data['corona_result']=='negative',0,1)
Covid_data['age_60_and_above']=np.where(Covid_data['age_60_and_above']=='No',0,1)
Covid_data['Male']=np.where(Covid_data['gender']=='male',1,0)
Covid_data['Female']=np.where(Covid_data['gender']=='female',1,0)
Covid_data['Contact_with_confirmed']=np.where(Covid_data['test_indication']=='Contact with confirmed',1,0)


# dropping column 'gender'
Covid_data.drop(columns=['gender'])


features=[]


#l1 regularization (lasso) for feature selection

#Creating new features to incorporate interactions
all_features=['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',  'age_60_and_above', 'Male', 'Female', 'Contact_with_confirmed']


num_features=len(all_features)

for i in range(num_features):
    for j in range(num_features):
        if j>i:
            interaction= '+'.join([all_features[i], all_features[j]])
            Covid_data[interaction]=Covid_data[all_features[i]]*Covid_data[all_features[j]]
            all_features.append(interaction)
   
    
print('size of features list', len(all_features))    

# storing data with new columns (features)
Covid_data.to_csv(main_file + "Covid_data.csv")



# assessing with all new features with interactions
X = Covid_data[all_features] # features
y = Covid_data['corona_result'] # labels


# dividing data into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=0)

# define the logistic regression model 
log_reg_model = LogisticRegression()


# fit the model with data
log_reg_model.fit(X_train,y_train)

# y prediction
y_pred=log_reg_model.predict(X_test)
   

# accuracy score logistic regression
score=accuracy_score(y_test, y_pred)
print('Full model accuracy score logistic regression:', score)

#Logistic regression model with l1 regularization and different penalties


# C = [i/100 for i in range(1,21)]

#C = [i/100 for i in range(21,31)]
Best_C = 0.12

#C = [10, 1, .1, .001]


# Best_C=C[0]
# max_acc=0

# for c in C:
#     logistic = LogisticRegression(penalty="l1", C=c, solver='liblinear').fit(X_train, y_train)
#     logistic.fit(X_train, y_train)
#     beta_vec=logistic.coef_[0]
#     print('C:', c)
#     print('Coefficient of each feature:', logistic.coef_)
#     print('Training accuracy:', logistic.score(X_train, y_train))
#     print('Test accuracy:', logistic.score(X_test, y_test))
#     print('')
#     if logistic.score(X_test, y_test)>=max_acc:
#         max_acc=logistic.score(X_test, y_test)
#         Best_C=c

logistic = LogisticRegression(penalty="l1", C=Best_C, solver='liblinear').fit(X_train, y_train)
logistic.fit(X_train, y_train)
print('C:', Best_C)
print('Coefficient of each feature:', logistic.coef_)
print('Training accuracy:', logistic.score(X_train, y_train))
print('Test accuracy of reduced model:', logistic.score(X_test, y_test))
print('')


# print('Best regularization parameter:', Best_C)
# print('Max accuracy:', max_acc)

#Creating new vector with remaining features
beta_vec=logistic.coef_[0]
index=np.where(beta_vec == 0.)[0]   
features_red = [i for i in all_features if all_features.index(i) not in index]


print(features_red)


# storing list of important features with interactions
with open("features.txt", "wb") as fp:  
    pickle.dump(features_red, fp)
