#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 07:24:08 2021

@author: stefanosbaros
"""


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
from sklearn.metrics import recall_score, precision_score, r2_score,mean_squared_error, accuracy_score
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
from numpy import loadtxt
import xgboost as xgb
from xgboost import XGBClassifier

# Our goal in this projest is to use logistic regression to predict
# whether someone with a certain profile of symptoms got infected with Covid-19 or not

# loading data
main_file="/Users/stefanosbaros/Desktop/Covid_ML_project/"
Covid_path = '/Users/stefanosbaros/Desktop/Covid_ML_project/corona_tested_individuals_ver_006.csv'
Covid_data = pd.read_csv(Covid_path)

Covid_data = Covid_data.fillna(value=np.nan)
Covid_data.replace(to_replace=['None'], value=np.nan, inplace=True)
Covid_data = Covid_data.dropna(axis=0)
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


#All features
all_features=['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',  'age_60_and_above', 'Male', 'Female', 'Contact_with_confirmed']


X = Covid_data[all_features] # features
y = Covid_data['corona_result'] # labels

# dividing data into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=0)

#Boosting classifier model 

Boost_model = XGBClassifier(n_estimators=1000, random_state=50, scale_pos_weight=28)
Boost_model.fit(X_train, y_train)

# y prediction
y_pred_bc=Boost_model.predict(X_test)

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



# Accuracy score logistic regression
Boost_model_score=Boost_model.score(X_test, y_test)
print('Accuracy score boosting model:', Boost_model_score)


# Precision score logistic regression
bc_pre_score=precision_score(y_test, y_pred_bc)
print('Precision score boosting model:', bc_pre_score)


# Recall score logistic regression
bc_score=recall_score(y_test, y_pred_bc)
print('Recall score boosting model:', bc_score)

# confusion matrix - logistic regression
cm_lr= metrics.confusion_matrix(y_test, y_pred_bc)
print(cm_lr)


plot_confusion_matrix(Boost_model , X_test, y_test)  
plt.show()  
