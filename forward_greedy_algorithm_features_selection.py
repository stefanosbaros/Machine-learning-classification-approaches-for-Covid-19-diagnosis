#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 07:48:32 2021

@author: stefanosbaros
"""


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel


# Our goal here is to find the best features i.e., symptoms that are the best predictors
# Covid-19 infection result

# The algorithm first assesses iteratively the performance of a logistic classifier with one feature, by considering all of them one by one
# and finds the best that gives the highest accuracy score. Once the best feature is found, the algorithm moves to check the performance of 
# classifier with two features by checking all possible combinations etc. That way, the features are ranked according to their significance.

# This algorithm does not guarantee that the best features are found in the right order but works well
# in practice


# loading data
main_file="/Users/stefanosbaros/Desktop/Covid_ML_project/"
Covid_path = '/Users/stefanosbaros/Desktop/Covid_ML_project/corona_tested_individuals_ver_006.csv'
Covid_data = pd.read_csv(Covid_path)


Covid_data.replace(to_replace=['None'], value=np.nan, inplace=True)
Covid_data = Covid_data.dropna(axis=0)
#print(Covid_data.info())
des1 = Covid_data[['cough','fever', 'sore_throat', 'shortness_of_breath', 'head_ache']].describe()
des2 = Covid_data[['cough','fever', 'sore_throat', 'shortness_of_breath', 'head_ache']].isna().sum().to_frame(name='missing').T
des=pd.concat([des1,des2])
print(des)

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
Covid_data[all_features] = Covid_data[all_features].astype('int64')


# forward greedy algorithm for features selection
i=1
best_features=[]
scores=[]

for i in range(9):
    best_feature=None
    max_score=0     
    for feature in all_features:
        features=best_features.copy()
        features.append(feature)
        
        # defining input and output data
        X = Covid_data[features] # features
        y = Covid_data['corona_result'] # labels
        
        
        # dividing data into train and test sets
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=40)
        
        
        # define the logistic regression model 
        log_reg_model = LogisticRegression()
        
        
        # fit the model with data
        log_reg_model.fit(X_train,y_train)
        
        # y prediction
        y_pred=log_reg_model.predict(X_test)
           
        
        
        # accuracy score logistic regression
        score=accuracy_score(y_test, y_pred)
        
        # recall score
        score2=recall_score(y_test, y_pred)
        
        print('Accuracy score:', score)
        print('Recall score:', score2)
        
        if score2>=max_score:
            max_score=score2
            best_feature=feature
                

    all_features.remove(best_feature)
    best_features.append(best_feature)
    scores.append(max_score)
    


print('Best features ranked from best to worst:', best_features)
print('Prediction accuracy with best features added one-by-one:', scores)
