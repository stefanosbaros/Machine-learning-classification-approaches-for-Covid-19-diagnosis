# A machine learning classification approach for Covid 19 diagnosis based on clinical symptoms

**Authors**: Stefanos Baros and Ana Jevtic (both authors contributed equally)

## Project description

In this project, we used data that were publicly reported by the Israeli Ministry of Health and developed two classifiers that can offer quick screening and efficient medical diagnosis for Covid-19. In particular, the developed classifiers were developed using logistic regression and random forests techniques and can predict whether an individual may be infected with Covid-19 based on several clinical symptoms such as cough, fever, shortness of breath etc. Our approach is motivated by the urgent and timely need for the development of online tools that can aid alleviating the burden on healthcare systems. We performed feature engineering and designed new features that build upon interaction effects to improve the performance of our classifiers. We also carried out model selection via l1 regularization to identify the important features. We evaluated the performance of our classifiers via standard accuracy metrics and confusion matrices.

. 

## Dataset description

The dataset used in this project originates from the [Covid tested individuals data set](https://github.com/nshomron/covidpred/tree/master/data) - which is downloaded from the Israeli Ministry of Health website and translated into English by the authors of the paper [Machine learning-based prediction of COVID-19 diagnosis based on symptoms](https://www.nature.com/articles/s41746-020-00372-6).

Data contains experiencing symptoms and Covid-19 test results for 1048575 people who were tested between 9th of October of 2020 and 11th of December of 2020 in Israel. There are 10 columns in the data, the first one recoding the Covid-19 test date, and the remaining ones recording symptoms (cough, fever, sore throat, shortness of breath, headache), gender, whether the individual came in contact with a confirmed Covid-19 positive case, whether the individual is above 60 years old or not and, finally the Covid-19 test result.



## Description of files

There are **three** main files in this project repository:

- `forward_greedy_algorithm_features_selection.py`
- `feature_engineering_and_l1_regularization_features_selection.py`
- `Covid_prediction_classification_approaches.py`


The `forward_greedy_algorithm_features_selection.py` implements a forward greedy selection algorithm for sorting features according to their importance. It works as follows. First, we start with an empty set of features and then gradually add one feature at a time to the set of selected features. We then apply our algorithm, in our case logistic regression, on the set of features by adding one at a time and obtain a different predictor every time whose accuracy is recorded. We update the list of selected features every time by choosing the features that yields the predictor with the smallest risk (error).


The `feature_engineering_and_l1_regularization_features_selection.py` is the file that contains our feature engineering and model selection approaches. By starting with the basic features in our data set we first design new features by considering all possible two-way interactions among them. Then, we perform l1 regularization to finally obtain a list with all features that are important.

The `Covid_prediction_classification_approaches.py.py` file contains the implementation of the logistic regression and random forest classification approaches for predicting whether an individual may be infected or not with Covid-19 based on the symptoms he is experiencing.

## Project details and results
First, we used one hot encoding to design new features that describe the Covid-19 test result, whether the individual is 60 years old and above, whether the individual came in contact with a confirmed Covid-19 positive case, and whether the individual is male or female. We then replaced related columns in the initial data set. Our methodology for constructing our classifiers can be classified in three main steps:

- Feature engineering for design of new features
- Model selection via l1 regularization 
- Design of logistic regression and random forest classifiers


### Feature engineering for design of new features

Initially the **features** in our model were:

- `features =['cough', 'fever', 'sore_throat', 'shortness_of_breath' 'head_ache','age_60_and_above', 'gender', 'test_indication']`
            
where `test_indication` captured whether the individual came in contact with confirmed positive case. After performing **one-hot encoding** our new list of **features** became:

- `features=['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',  'age_60_and_above', 'Male', 'Female', 'Contact_with_confirmed']`

By considering all possible **interactions among these main features** except the male-female one we obtained the following list of **features**:

- `features=['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache', 'age_60_and_above', 'Male', 'Female', 'Contact_with_confirmed', 'cough+fever', 'cough+sore_throat', 'cough+shortness_of_breath', 'cough+head_ache', 'cough+age_60_and_above', 'cough+Male', 'cough+Female', 'cough+Contact_with_confirmed', 'fever+sore_throat', 'fever+shortness_of_breath', 'fever+head_ache', 'fever+age_60_and_above', 'fever+Male', 'fever+Female', 'fever+Contact_with_confirmed', 'sore_throat+shortness_of_breath', 'sore_throat+head_ache', 'sore_throat+age_60_and_above', 'sore_throat+Male', 'sore_throat+Female', 'sore_throat+Contact_with_confirmed', 'shortness_of_breath+head_ache', 'shortness_of_breath+age_60_and_above', 'shortness_of_breath+Male', 'shortness_of_breath+Female', 'shortness_of_breath+Contact_with_confirmed', 'head_ache+age_60_and_above', 'head_ache+Male', 'head_ache+Female', 'head_ache+Contact_with_confirmed', 'age_60_and_above+Male', 'age_60_and_above+Female', 'age_60_and_above+Contact_with_confirmed', 'Male+Contact_with_confirmed', 'Female+Contact_with_confirmed']`

This list of features was subsequently used in model selection.

### Model selection via l1 regularization

Having obtained a comprehensive list of features including new ones that capture interactions among the main features we performed logistic regression with l1 regularization in order to reduce this list to one containing only the important features. By defining the regularization penalty as l=1/C, we first performed logistic regression with l1 regularization by considering values for the constant C as given in the table below. For each value C, we obtained a different logistic regression classifier whose performance is recorded. 


| C | Accuracy on training set | Accuracy on test set |
| ----------- | ----------- | ----------- | 
| 10 | 0.9071 | 0.9062 | 3.0579 |
| 1 | 0.6307 | 13.5849 | 2.9827 |
| 0.1 | 0.6483 | 12.9366 | 2.9064 |
|  0.01 | 0.6853 | 11.5732 | 2.7271 |


From the above table, we see that C=0.1 yields the predictor with the best accuracy both on training and test sets. Given that, we then repeat this process but now considering values for the constant C around 0.1. Specifically, we consider values for C in the range [0-0.3] with 0.01 step. The best performance is obtained for **C=0.12** which corresponds to **regularization penalty l=8.33**.



### Discussion on the results
A couple of things are important to notice here. First, Model 2 performs better than Model 1, which means that a cubic function explains the data better than a quadratic one. Better performance here means larger Adjusted R-squared and smaller MSE and MAE. Model 3 and Model 4 which include the month and the hour as features lead to better performance than Model 2. This verifies our earlier argument that the month and hour are important variables that affect the electric load. Lastly, Model 5, 6 and the full model take into account the interaction effects and lead to further improvements in performance. Lastly, we see that the full model that includes all interaction effects leads to the best overall performance compared with the remaining six models i.e., largest Adjusted R-squared and smallest MSE and MAE. 

![Actual vs predicted load](actual_predicted_load_temp.png) ![Actual vs predicted load](actual_predicted.png)
        
