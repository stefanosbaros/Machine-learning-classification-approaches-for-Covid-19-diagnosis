#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 07:54:08 2021

@author: Stefanos Baros and Ana Jevtic

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
import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



# Our goal in this projest is to use logistic regression to predict
# whether someone with a certain profile of symptoms got infected with Covid-19 or not

# loading data

# Load features from file "feature_engineering_and_l1_regularization_features_selection"
with open("features.txt", "rb") as fp:   # Unpickling
    all_features= pickle.load(fp)
    

Covid_path = '/Users/stefanosbaros/Desktop/Covid_ML_project/Covid_data.csv'
Covid_data = pd.read_csv(Covid_path)


#all_features=['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',  'age_60_and_above', 'Male', 'Female', 'Contact_with_confirmed']


# assessing with all new features with interactions
X = Covid_data[all_features] # features
y = Covid_data['corona_result'] # labels

# dividing data into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=50)

#define the logistic regression model 
log_reg_model = LogisticRegression()

# weights = {0:0.095, 1:1.0}
# log_reg_model = LogisticRegression(solver='lbfgs', class_weight=weights)

# fit the model with data
log_reg_model.fit(X_train,y_train)

# y prediction
y_pred_lr=log_reg_model.predict(X_test)

# probs=log_reg_model.predict_proba(X_test)

# probs = probs[:, 1]

# auc = roc_auc_score(y_test, probs)
# print('AUC: %.2f' % auc)


# def plot_roc_curve(fpr, tpr):
#     plt.plot(fpr, tpr, color='orange', label='ROC')
#     plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend()
#     plt.show()

# fnr, tnr, thresholds = roc_curve(y_test, probs)


# plot_roc_curve(fnr, tnr)


# accuracy score logistic regression
lr_score=accuracy_score(y_test, y_pred_lr)
print('Accuracy score logistic regression:', lr_score)

# confusion matrix - logistic regression
cm_lr= metrics.confusion_matrix(y_test, y_pred_lr)
print(cm_lr)

# Random forest classifier

# Instantiate model with 1000 decision trees


rf = RandomForestClassifier(n_estimators=100,  random_state=50, criterion='entropy', max_depth=10, bootstrap=True, class_weight={0:1,1:13})

#rf = BalancedRandomForestClassifier(n_estimators=500)


# Train the model on training data
rf.fit(X_train, y_train);

# Use the forest's predict method on the test data
y_pred_rf = rf.predict(X_test)

# accuracy score random forest
rf_score=accuracy_score(y_test, y_pred_rf)
print('Random forest accuracy score:', rf_score)

# confusion matrix - random forest
cm_rf= metrics.confusion_matrix(y_test, y_pred_rf)
print(cm_rf)


plot_confusion_matrix(rf, X_test, y_test)  
plt.show()  

plot_confusion_matrix(log_reg_model , X_test, y_test)  
plt.show()  
