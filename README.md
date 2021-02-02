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

- `features = ['cough', 'fever', 'sore_throat', 'shortness_of_breath' 'head_ache','age_60_and_above', 'gender', 'test_indication']`
            
where `test_indication` captured whether the individual came in contact with confirmed positive case. After performing **one-hot encoding** our new list of **features** became:

- `features = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',  'age_60_and_above', 'Male', 'Female', 'Contact_with_confirmed']`

By considering all possible **interactions among these main features** except the male-female one we obtained the following list of **features**:

- `features = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache', 'age_60_and_above', 'Male', 'Female', 'Contact_with_confirmed', 'cough+fever', 'cough+sore_throat', 'cough+shortness_of_breath', 'cough+head_ache', 'cough+age_60_and_above', 'cough+Male', 'cough+Female', 'cough+Contact_with_confirmed', 'fever+sore_throat', 'fever+shortness_of_breath', 'fever+head_ache', 'fever+age_60_and_above', 'fever+Male', 'fever+Female', 'fever+Contact_with_confirmed', 'sore_throat+shortness_of_breath', 'sore_throat+head_ache', 'sore_throat+age_60_and_above', 'sore_throat+Male', 'sore_throat+Female', 'sore_throat+Contact_with_confirmed', 'shortness_of_breath+head_ache', 'shortness_of_breath+age_60_and_above', 'shortness_of_breath+Male', 'shortness_of_breath+Female', 'shortness_of_breath+Contact_with_confirmed', 'head_ache+age_60_and_above', 'head_ache+Male', 'head_ache+Female', 'head_ache+Contact_with_confirmed', 'age_60_and_above+Male', 'age_60_and_above+Female', 'age_60_and_above+Contact_with_confirmed', 'Male+Contact_with_confirmed', 'Female+Contact_with_confirmed']`

This list of features was subsequently used in model selection.

### Model selection via l1 regularization

Having obtained a comprehensive list of features including the new ones that capture interactions among the main features we performed logistic regression with l1 regularization in order to reduce this list to one containing only the important features. By defining the regularization penalty as l=1/C, we first performed logistic regression with l1 regularization by considering values for the constant C as given in the table below. For each value C, we obtained a different logistic regression classifier whose performance on the training set and test set is recorded. We note that, **10% of the data is used for validation and 90% for training**.


| C | Accuracy on training set | Accuracy on test set |
| ----------- | ----------- | ----------- | 
| 10 | 0.90713 | 0.90623 | 
| 1 | 0.90714 | 0.90624 | 
| 0.1 | 0.90715 | 0.90627| 
|  0.001 | 0.90186 | 0.90041 | 


From the above table, we see that **C=0.1** yields the predictor with the **best accuracy** both on training and test sets. Given that, we then repeat this process but now considering values for the constant C around 0.1. Specifically, we consider values for C in the range [0-0.3] with 0.01 step. The **best** performance, training accuracy=0.907154, test accuracy= 0.906273 is obtained for **C=0.12** which corresponds to **regularization penalty l=8.33**.  For this penalty factor, we obtained the following vector with the logistic regression coefficients:

`beta = [ 1.32326951  2.16469435  3.26884738  2.9581783   4.16169908  0.`
 `-0.8879601  -1.11242282  1.52275301 -1.1211198  -0.46753865 -0.17598219`
  `-0.34580276  0.66996671  0.06875167  0.         -0.7048826  -0.2161716`
  `-0.57902847 -0.96274254  0.44389867  0.1993682   0.         -1.03685355`
  `-0.55990202 -1.54027642  0.08103512  0.          0.         -1.64762446`
  `-0.72580185  0.          0.          0.06835987 -1.51138044  0.15431912`
   `0.          0.04212707 -2.59252317 -0.12636422 -0.15799593  0.03410816`
   `0.          1.01950092  1.2016845 ]`



By retaining only the features which correspond to **nonzero** coefficients we finally ended up with the following list of **important features**:

- `features = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache', 'Male', 'Female', 'Contact_with_confirmed', 'cough+fever', 'cough+sore_throat', 'cough+shortness_of_breath', 'cough+head_ache', 'cough+age_60_and_above', 'cough+Male', 'cough+Contact_with_confirmed', 'fever+sore_throat', 'fever+shortness_of_breath', 'fever+head_ache', 'fever+age_60_and_above', 'fever+Male', 'fever+Contact_with_confirmed', 'sore_throat+shortness_of_breath', 'sore_throat+head_ache', 'sore_throat+age_60_and_above', 'sore_throat+Contact_with_confirmed', 'shortness_of_breath+head_ache', 'shortness_of_breath+Female', 'shortness_of_breath+Contact_with_confirmed', 'head_ache+age_60_and_above', 'head_ache+Female', 'head_ache+Contact_with_confirmed', 'age_60_and_above+Male', 'age_60_and_above+Female', 'age_60_and_above+Contact_with_confirmed', 'Male+Contact_with_confirmed', 'Female+Contact_with_confirmed']`


### Design of logistic regression and random forest classifiers

Lastly, we designed a logistic regression classifier and a random forest classifier by training our algorithm on 90% of our data and keeping the remaining 10% for validation. The random forest classifier is designed using **1000 ensembled trees constructed from Bootstrap samples based on the entropy information gain criterion and with maximum depth 10** (n_estimators=1000, random_state=50, criterion='entropy', max_depth=10, bootstrap=True). We varied the depth of the trees from 4 to 15 in order to consider more complex classes for our classifiers. Beyond depth 10, we haven't seen improvement in the the performance of our random forest. Below we present the **performance of our logistic regression and random forest classifiers** which is **very close**. 

| Logistic regression accuracy | Random forest accuracy |
| ----------- | ----------- |
| 0.910078 | 0.910269|

We also show the confusion matrices of both classifiers.


        
