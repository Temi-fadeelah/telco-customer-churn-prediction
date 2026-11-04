
# Telecommunications Customer Churn Prediction

## 1. Project Overview

This project uses machine learning techniques to predict customer churn in a telecommunications company.

Customer churn occurs when a customer stops using a company's services. Predicting churn can help telecommunications companies identify customers who may leave and develop strategies to improve customer retention.

The project uses the IBM Telco Customer Churn dataset to explore customer behaviour, build classification models and evaluate their ability to predict churn.

## 2. Problem Statement

Telecommunications companies face challenges retaining customers. This project aims to develop a machine learning model that predicts whether a customer is likely to leave the company based on their service usage, account information and other customer characteristics.

The objective is to compare different classification algorithms and identify a model that provides useful predictions.

## 3. Dataset Description

The dataset used is the IBM Telco Customer Churn dataset.

- Number of records: Approximately 7,043 customers.
- Target variable: `Churn`.
- Target values: Yes (customer churned) and No (customer stayed).

The dataset contains information about:
- Customer demographics.
- Account information.
- Services subscribed to.
- Contract types.
- Monthly and total charges.
- Customer tenure.

The `customerID` column was excluded from predictive features because it is an identifier rather than a meaningful customer characteristic.

## 4. Data Cleaning and Preparation

The following steps were performed:

1. Inspected dataset columns, data types and missing values.
2. Checked for duplicate records.
3. Identified blank entries in `TotalCharges`.
4. Converted `TotalCharges` to a numeric data type.
5. Handled blank `TotalCharges` entries by replacing them with zero, after considering their relationship with customer tenure.
6. Identified `Churn` as the target variable.
7. Removed `customerID` from the input features.

Preprocessing decisions were implemented using Scikit-learn pipelines.

## 5. Exploratory Data Analysis (EDA)

Exploratory data analysis was conducted to understand customer churn patterns.

The analysis examined:

- Overall customer churn rate.
- Churn across different contract types.
- Relationship between customer tenure and churn.
- Relationship between monthly charges and churn.

Visualisations were created to communicate the patterns observed in the dataset.

**Key findings:** [Insert your actual findings from the notebook.]

## 6. Data Preprocessing

The dataset was divided into input features and the target variable.

The following preprocessing techniques were applied:

- One-hot encoding for categorical variables.
- Standard scaling for numerical variables.
- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- Stratified train/test splitting to maintain the target class distribution.

Preprocessing was included inside Scikit-learn pipelines to prevent data leakage. The preprocessing steps were fitted using training data only.

## 7. Model Development

The following classification algorithms were considered:

### Logistic Regression

Used as the baseline classification model.

### Random Forest

Used as a comparison model to evaluate whether an ensemble of decision trees could improve predictive performance.

### Multi-Layer Perceptron (Optional)

A small neural network that may be explored as an extension.

## 8. Model Validation and Selection

Stratified five-fold cross-validation was used to compare model performance on the training data.

The final test set was kept separate and reserved for evaluating the selected model.

ROC-AUC was used as the initial model selection criterion, alongside precision, recall and F1-score.

## 9. Model Evaluation

The selected model was evaluated using the following metrics:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

**Final model:** [Logistic Regression]

**Evaluation results:**

| Metric | Score |
|---|---|
| Precision | [0.6313] |
| Recall | [0.5401] |
| F1-score | [ 0.5821] |
| ROC-AUC | [0.8403] |

The confusion matrix was used to examine correct and incorrect predictions.

Accuracy was not used as the only measure of model performance because churn prediction involves an imbalance between customers who leave and those who stay.

## 10. Business Interpretation

False negatives represent customers who actually churned but were predicted to stay. These errors may cause the company to miss opportunities for customer retention.

False positives represent customers predicted to churn who actually stayed. These predictions may lead to unnecessary retention spending.

The company must consider the financial implications of both types of errors when using the model.

## 11. Feature Importance
The Random Forest model identified tenure and monthly charges as the two most important features. Contract type, payment method, and online security also contributed to its predictions.

These findings show associations in the dataset and should not be interpreted as proof of direct causation.
## 12. Business Recommendations

Based on the patterns identified in the analysis, potential strategies include:

1. Improving onboarding and customer support for new customers.
2. Investigating incentives for customers on month-to-month contracts.
3. Reviewing service packages and pricing where monthly charges are associated with churn.
4. Improving technical support and service reliability.
5. Using churn predictions to identify customers who may benefit from additional support.

These recommendations should be tested before being treated as proven solutions.

## 13. Limitations

- The model is trained on historical customer data.
- Customer behaviour may change over time.
- Predictive associations do not establish causation.
- The model may incorrectly classify some customers.
- Model performance depends on the quality and representativeness of the dataset.

## 14. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab
- Joblib

## 15. Project Files

The project contains the following outputs:

- `README.md` – Project documentation.
- `churn_prediction_model.ipynb` – Notebook containing the analysis and model development.
- `churn_prediction_model.pkl` – Saved trained model.
- `model_evaluation_metrics.csv` – Model evaluation results.
- `confusion_matrix.png` – Confusion matrix visualisation.
- `feature_importance.png` – Feature importance visualisation.
- `model_comparison.png` – Model performance comparison.

## 16. Installation and Setup

Install the required libraries using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

If using the optional Streamlit application, install Streamlit:

```bash
pip install streamlit
```

## 17. Reproducibility

To reproduce the project:

1. Download or obtain the IBM Telco Customer Churn dataset.
2. Open the project notebook in Google Colab or Jupyter Notebook.
3. Upload the dataset to the notebook environment.
4. Run the notebook cells sequentially, starting with data loading and cleaning.
5. Perform exploratory data analysis and preprocessing.
6. Train and validate the classification models.
7. Evaluate the selected model using the test dataset.

The notebook contains the code required to reproduce the analysis and model training.

## 18. Conclusion

This project demonstrates how machine learning classification techniques can be applied to telecommunications customer churn prediction.

By analysing customer information and comparing classification algorithms, the project provides an approach to identifying customers who may be at risk of leaving and supporting data-informed customer retention strategies.
