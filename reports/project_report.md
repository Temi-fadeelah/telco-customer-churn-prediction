# Telecommunications Customer Churn Prediction

## 1. Introduction

This project uses machine learning to predict customer churn in the telecommunications industry. Customer churn occurs when customers discontinue their services or switch to another provider.

The aim is to identify customers who are more likely to leave so that telecommunications companies can develop better customer retention strategies.

## 2. Dataset and Data Preparation

The project uses the IBM Telco Customer Churn dataset, containing approximately 7,043 customer records.

The dataset includes customer demographics, account information, subscribed services, tenure, monthly charges and churn status.

The following preparation steps were carried out:

- Inspected the dataset for missing values, data types and duplicates.
- Converted `TotalCharges` into a numerical column.
- Replaced blank `TotalCharges` values with zero, corresponding to customers with zero tenure.
- Removed the customer ID column because it does not provide useful predictive information.
- Converted the churn target into binary values: 1 for Yes and 0 for No.
- Used one-hot encoding for categorical variables and standardisation for numerical variables.

## 3. Exploratory Data Analysis

Exploratory data analysis was performed to understand customer churn patterns.

The analysis examined:
- Overall customer churn distribution.
- Churn across different contract types.
- Churn across customer tenure groups.
- The relationship between monthly charges and churn.

Visualisations were created to communicate these patterns.

The Random Forest feature importance analysis also highlighted tenure, monthly charges, contract type, payment method and online security as important features in the model's predictions.

## 4. Model Development

Two classification algorithms were developed:

1. Logistic Regression
2. Random Forest

Both models were built using Scikit-learn pipelines containing preprocessing and classification steps.

The dataset was divided into 80% training data and 20% testing data using stratified sampling.

Five-fold stratified cross-validation was used to compare model performance and select the final model.

## 5. Model Comparison

The cross-validation results were:

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6497 | 0.5411 | 0.5898 | 0.8455 |
| Random Forest | 0.6378 | 0.4569 | 0.5322 | 0.8242 |

Logistic Regression performed better across all four evaluation metrics and was selected as the final model.

## 6. Final Model Evaluation

The selected Logistic Regression model was evaluated on the held-out test dataset.

| Metric | Score |
|---|---:|
| Precision | 0.6313 |
| Recall | 0.5401 |
| F1-score | 0.5821 |
| ROC-AUC | 0.8403 |

The ROC-AUC score of approximately 0.84 indicates good ability to distinguish between customers who churn and those who do not.

However, the recall score of approximately 54% shows that the model still misses some customers who actually churn.

## 7. Key Findings

The analysis identified the following important predictive features:

- Customer tenure.
- Monthly charges.
- Month-to-month contracts.
- Electronic check payment method.
- Lack of online security services.

Tenure and monthly charges were the two most important features in the Random Forest feature importance analysis.

These findings indicate associations within the dataset and do not establish that these factors directly cause customer churn.

## 8. Business Recommendations

Based on the findings, telecommunications companies could consider:

- Developing customer engagement programmes for newer customers.
- Reviewing pricing and service packages to improve customer satisfaction.
- Offering suitable incentives for customers considering longer-term contracts.
- Improving awareness and accessibility of online security services.
- Using churn predictions to identify customers who may benefit from additional support.

Any retention strategy should be tested to determine whether it actually reduces churn.

## 9. Limitations

The project has several limitations:

- The model was developed using one telecommunications dataset.
- Logistic Regression did not identify every customer who actually churned.
- Feature importance indicates predictive associations, not causation.
- Model performance may change when applied to customers from another company or time period.
- Further tuning and evaluation may improve performance.

## 10. Conclusion

This project demonstrated how machine learning can be used to analyse telecommunications customer churn and predict customers who may discontinue their services.

Logistic Regression was selected over Random Forest based on its cross-validation performance. It achieved a test ROC-AUC of 0.8403.

The findings demonstrate the potential of predictive analytics to support customer retention decisions, while highlighting the need for further improvements in identifying customers at risk of leaving.
