#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:19:47 2021

@author: stefanosbaros
"""


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
from sklearn.metrics import recall_score, r2_score,mean_squared_error, accuracy_score
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


features_imp=['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache','Contact_with_confirmed']
num_features=len(features_imp)

for i in range(num_features):
    features_red=features_imp.copy()
    features_red.pop(i)
    interaction= '-'.join(features_red)
    all_features.append(interaction)
    Covid_data[interaction]=1
    for j in range(num_features):
       if j!=i:
           prod=Covid_data[interaction]*Covid_data[features_imp[j]]
           Covid_data[interaction]=prod
           
           
print(all_features)

with open("features_five_deg.txt", "wb") as fp:  
    pickle.dump(all_features, fp)       
    
print(all_features)
# storing data with new columns (features)
Covid_data.to_csv(main_file + "Covid_data_five_deg_interactions.csv")


# assessing with all new features with interactions
X = Covid_data[all_features] # features
y = Covid_data['corona_result'] # labels


# dividing data into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=0)

#logistic regression model 
log_reg_model = LogisticRegression()

#logistic regression model with weights
#weights = {0:0.039, 1:1.0}
#log_reg_model = LogisticRegression(solver='lbfgs', class_weight=weights)


# fit the model with data
log_reg_model.fit(X_train,y_train)

# y prediction
y_pred=log_reg_model.predict(X_test)
   

# recall score
print('Recall score:', recall_score(y_test, y_pred))


# accuracy score logistic regression
score=accuracy_score(y_test, y_pred)
print('Full model accuracy score logistic regression:', score)


cm_lr= metrics.confusion_matrix(y_test, y_pred)
print(cm_lr)


#Logistic regression model with l1 regularization and different penalties

#C = [i/100 for i in range(1,21)]

#C = [i/100 for i in range(21,31)]
C = [0.01]

#C = [10, 1, .1, 0.01, .001]


Best_C=C[0]
max_acc=0

for c in C:
    weights = {0:0.04, 1:1.0}
    logistic = LogisticRegression(penalty="l1", C=c, solver='liblinear', class_weight=weights).fit(X_train, y_train)
    logistic.fit(X_train, y_train)
    y_pred=logistic.predict(X_test)
    beta_vec=logistic.coef_[0]
    print('C:', c)
    print('Coefficient of each feature:', logistic.coef_)
    print('Recall score:', recall_score(y_test, y_pred))
    print('Test accuracy:', logistic.score(X_test, y_test))
    print('')
    if recall_score(y_test, y_pred)>max_acc:
        max_acc=recall_score(y_test, y_pred)
        y_pred_best=y_pred
        Best_C=c

weights = {0:0.04, 1:1.0}
logistic = LogisticRegression(penalty="l1", C=Best_C, solver='liblinear', class_weight=weights).fit(X_train, y_train)
logistic.fit(X_train, y_train)
y_pred_best=logistic.predict(X_test)
print('C:', Best_C)
print('Coefficient of each feature:', logistic.coef_)
print('Recall score:', recall_score(y_test, y_pred_best))
print('Test accuracy of reduced model:', logistic.score(X_test, y_test))
print('')


print('Best regularization parameter:', Best_C)
print('Max accuracy:', max_acc)

#Creating new vector with remaining features
beta_vec=logistic.coef_[0]
index=np.where(beta_vec == 0.)[0]   
features_red = [i for i in all_features if all_features.index(i) not in index]


print(features_red)


# storing list of important features with interactions
with open("features_five_deg.txt", "wb") as fp:  
    pickle.dump(features_red, fp)
