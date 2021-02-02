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

## Project details
###

## Results

The **features** used to construct a regression model are:

- `features =['TREND','TMP','TMP2','TMP3','SIN_MONTH','COS_MONTH','TMPxSIN_MONTH',`
            `'TMPxCOS_MONTH','TMP2xSIN_MONTH','TMP2xCOS_MONTH', 'TMP3xSIN_MONTH','TMP3xCOS_MONTH',`
           ` 'TMPxSIN_HOUR','TMPxCOS_HOUR','TMP2xSIN_HOUR','TMP2xCOS_HOUR', 'TMP3xSIN_HOUR','TMP3xCOS_HOUR',`
           ` 'DTMPxSIN_HOUR', 'DTMPxCOS_HOUR', 'Holi','isWknd']`
            
where, TMP, TMP2 and TMP3 denote the temperature, temperature squared and temperature in the cubic power, respectively. The features SIN_MONTH, COS_MONTH, SIN_HOUR, COS_HOUR denote the cyclical variables associated with the particular month of the year and hour of the day. 

We construct **seven regression models** including the full model, each time considering a different **subset of features**:

- `Model 1= ['TREND','TMP','TMP2']`
- `Model 2= ['TREND','TMP','TMP2','TMP3'])`
- `Model 3= ['TREND','TMP','TMP2','TMP3','SIN_MONTH','COS_MONTH'])`
- `Model 4= ['TREND','TMP','TMP2','TMP3','SIN_MONTH','COS_MONTH', 'DTMPxSIN_HOUR', 'DTMPxCOS_HOUR'])`
- `Model 5=['TREND','TMP','TMP2','TMP3','SIN_MONTH','COS_MONTH','TMPxSIN_MONTH',`
            `'TMPxCOS_MONTH','TMP2xSIN_MONTH','TMP2xCOS_MONTH', 'TMP3xSIN_MONTH','TMP3xCOS_MONTH'])`
- `Model 6= ['TREND','TMP','TMP2','TMP3','SIN_MONTH','COS_MONTH','TMPxSIN_MONTH',`
            `'TMPxCOS_MONTH','TMP2xSIN_MONTH','TMP2xCOS_MONTH', 'TMP3xSIN_MONTH','TMP3xCOS_MONTH','DTMPxSIN_HOUR', 'DTMPxCOS_HOUR']`
- `Full model= ['TREND','TMP','TMP2','TMP3','SIN_MONTH','COS_MONTH','TMPxSIN_MONTH',`
            `'TMPxCOS_MONTH','TMP2xSIN_MONTH','TMP2xCOS_MONTH', 'TMP3xSIN_MONTH','TMP3xCOS_MONTH',`
           ` 'TMPxSIN_HOUR','TMPxCOS_HOUR','TMP2xSIN_HOUR','TMP2xCOS_HOUR', 'TMP3xSIN_HOUR','TMP3xCOS_HOUR',`
           ` 'DTMPxSIN_HOUR', 'DTMPxCOS_HOUR', 'Holi','isWknd']`

The performance of these different models is assessed using **standard goodness-of-fit and accuracy criteria** as shown in the table below. 


| Model | Adjusted R-squared | Mean squared error (MSE) | Mean absolute error (MAE) |
| ----------- | ----------- | ----------- | ----------- |
| Model 1 | 0.6219 | 13.9104 | 3.0579 |
| Model 2 | 0.6307 | 13.5849 | 2.9827 |
| Model 3 | 0.6483 | 12.9366 | 2.9064 |
| Model 4 | 0.6853 | 11.5732 | 2.7271 |
| Model 5 | 0.7117 | 10.6013 | 2.6510 |
| Model 6 | 0.7404 | 9.5481 | 2.4757 |
| Full model | 0.7951 | 7.5349 | 2.2172 |


### Discussion on the results
A couple of things are important to notice here. First, Model 2 performs better than Model 1, which means that a cubic function explains the data better than a quadratic one. Better performance here means larger Adjusted R-squared and smaller MSE and MAE. Model 3 and Model 4 which include the month and the hour as features lead to better performance than Model 2. This verifies our earlier argument that the month and hour are important variables that affect the electric load. Lastly, Model 5, 6 and the full model take into account the interaction effects and lead to further improvements in performance. Lastly, we see that the full model that includes all interaction effects leads to the best overall performance compared with the remaining six models i.e., largest Adjusted R-squared and smallest MSE and MAE. 

![Actual vs predicted load](actual_predicted_load_temp.png) ![Actual vs predicted load](actual_predicted.png)
        
