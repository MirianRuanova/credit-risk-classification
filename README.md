# credit-risk-classification
## Overview of the Analysis

The purpose of the analysis is to identify the creditworthiness of the loan borrowers applying Machine Learning.
In order to do that, we have used a dataset of historical lending activity from a peer-to-peer lending services company to build the supervised learning model.
The supervised learning model takes a set of known answers called labels (*healthy credit*/*risky credit*) and fits a model with a set of features (inputs, each attribute on the columns) that correspond to the labels. It required us to feed the correct answers to the model (`loan_status` column)
The method applied was Logistic Regression, a statistical method for predicting binary outcomes from the data. In other words, instead of forecasting quantitative numbers, it classifies the type of loans using a binary (true positive/true negative) approach to predict membership in a category. 
The goal of the algorithm is to estimate the probability that an instance belongs to a particular class. It assumes a linear relationship between the input features and the log odds of the target variable being in a specific class. The log odds are then transformed using the logistic function (sigmoid function) to obtain the predicted probabilities. 
The stages of the machine learning process that have been addressed as part of the analysis are:
1. Data Preparation:
  1.1 Separeta the data into y, the binary target variable, consisting of two classes denoted as 0 (healthy credit) and 1 (risky credit)..
  1.2 X is crerated as the variable that contents the features used to train the machine learning models.These features provide important information about the loan applicants and their financial profiles. Specifically, the features include `loan_size`, `interest_rate`, `borrower_income`, `debt_to_income`, `num_of_accounts`, `derogatory_marks`, and `total_debts`. Each feature captures a specific aspect related to the borrower's creditworthiness.
2. Model training: 
  2.1 The dataset is split into separate training and testing sets using the `train_test_split `function. This step ensures that the model is trained on a portion of the data and evaluated on unseen data.
    2.1.1 `X_train` represents the training set features, which are used to teach the model patterns and relationships.
    2.1.2 `X_test `corresponds to the testing set features, used to assess the model's performance.
    2.1.3 `y_train` contains the training set labels, representing the known loan status for the corresponding training instances.
    2.1.4 `y_test` holds the testing set labels, used for evaluating the model's predictions.
  2.2 The Logistic Regression model is instantiated. Logistic Regression is a statistical algorithm that estimates the parameters (coefficients) of a linear equation by maximizing the likelihood function or minimizing a cost function. In this case, it's used to predict the loan status based on the input features.
  2.3 After, it time to fit the model which means using the training data to estimate or learn the parameters of the chosen machine learning algorithm.The `.fit` function takes the input features (`X_train`) and corresponding target labels (`y_train)` as input and learns the underlying patterns or relationships between the input features and the target variable.
3. Prediction: 
  3.1 Once the model is trained and the coefficients are estimated, it can be used to make predictions on new, unseen data.
  3.2 For a given set of input features, the model calculates the log-odds (logistic function) of the target variable being in the positive class (risky credit).
  3.3 The log-odds are transformed into probabilities using the logistic (sigmoid) function, which maps the range from negative infinity to positive infinity to the range between 0 and 1
  3.4 Based on the threshold (usually 0.5),the predicted probability is classified as either 0 (healthy credit) or 1 (risky credit), corresponding to the predicted class.

However, after separating the data into two variables (y and X) and checking the balance of the target values using the function `value_counts()`, it seems the total of *Healthy Loans* class (75,036) in the dataset is significantly higher than the *Hight Risk Loan* class (2,500). To address class imbalance in the dataset we used the `RandomOverSampler` technique.

`RandomOverSampler` is implemented in the imbalanced-learn library, which provides various methods for handling imbalanced datasets. It aims to balance the class distribution by randomly oversampling the minority class (the class with fewer instances) until it reaches the same size as the majority class. In the example, after implemented `RandomOverSampler` the values of each class is 56,277. 


## Results

* Machine Learning Model 1: Logistic Regressin Model with the Original Data.

  - A *balanced accuracy* of *0.9442676901753825*.
  That indicates that the model is performing well in terms of overall accuracy. It suggests that, on average, the model is correctly classifying approximately 94.43% of instances across all classes, taking into account both *Healthy Loan* and *Hight Risk Loan* classes. This is a good performance and indicates that the model is capable of making accurate predictions.

  *Healthy Loan*

    - For the class *Healthy Loan*, the precision, recall, and F1-score is *1* which indicate that the model performs perfectly in predicting instances of the *Healthy Loan* class. 

    - *Precision* of *1* means that all the instances predicted as *Healthy Loan* are indeed true positives. 
    - *Recall* of *1* means that the model correctly identifies all the actual *Healthy Loan* instances. 
    - The *F1-score* of *1* is the harmonic mean of precision and recall, indicating a perfect balance between the two metrics.
  
  *High-Risk Loan*
    - For the other class, *High-Risk Loan*, the *precision* of *0.87* means that out of all the instances predicted as *High-Risk Loan*,  approximately 87% are true positives, while 13% are false positives. 
    
    - The *recall* of *0.89* suggests that the model correctly identifies 89% of the actual "High-Risk Loan" instances. 
    
    - The *F1-score* of *0.88* represents the weighted average of precision and recall, accounting for their balance.

  In summary, the model shows excellent performance in predicting the "Healthy Loan" class, achieving perfect precision, recall, and F1-score. For the "High-Risk Loan" class, the model has good precision, recall, and F1-score, indicating that it performs well in identifying instances of that class, although there is some room for improvement in correctly classifying all "High-Risk Loan" instances.

* Machine Learning Model 2:Logistic Regression Model with Resampled Training Data

  Using Logistic Regression Model with Resamped Training Data we observed an increase in the *balanced accuracy score* of *0.994180571103648* and *precision*, *recall*, and *F1-scores* of *0.99* for both classes indicate a significant improvement in the performance of the logistic regression model when applied to the resampled training data. 

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

1. Balanced Accuracy Score:

  - Original Data: The *balanced accuracy score* was *0.9442676901753825*.

  - Resampled Data: The *balanced accuracy score* increased to *0.994180571103648*.

  The balanced accuracy score represents the overall accuracy considering the class imbalance. The higher the score, the better the model performs. In this case, the model trained on the resampled data achieves a much higher balanced accuracy score, indicating its improved ability to correctly classify instances from both classes.

2. Precision, Recall, and F1-score:

  2.1 Original Data:
  - For the *Healthy Loan* class, the precision, recall, and F1-score were 1.
  - For the *High-Risk Loan* class, the *precision* was *0.87*, *recall* was *0.89*, and *F1-score* was *0.88*.
  2.2 Resampled Data:
  - For both the *Healthy Loan* and *High-Risk Loan* classes, the *precision*, *recall*, and *F1-score* are now *0.99*.
  - Precision represents the ability of the model to correctly identify true positives, recall measures the model's ability to find all positive instances, and the F1-score is the harmonic mean of precision and recall. The higher the values, the better the model performs in terms of correctly identifying instances from both classes. 

Finally, I would recommend the resampled model since it shows a significant improvement in the balanced accuracy score, precision, recall, and F1-scores for both classes compared to the original model. The improving the performance of the resampled model is due to addressing the class imbalance issue in the dataset which allows us to reach the perfect scores of 0.99 for both classes which indicates that the model trained on the resampled data is highly accurate and effective in predicting instances from both classes.