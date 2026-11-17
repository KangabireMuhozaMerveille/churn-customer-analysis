# Customer Churn Analysis & Prediction

## Project Overview

Customer churn is a major challenge for subscription-based businesses. When customers discontinue a service, businesses lose recurring revenue and may incur additional costs acquiring new customers.

This project explores customer churn in a telecommunications company using exploratory data analysis and machine learning. The objective is to identify patterns associated with customer churn and develop classification models that can help businesses identify customers who may be at risk of leaving.

## Objectives

- Explore customer demographics, account information and service subscriptions.
- Identify patterns and characteristics associated with customer churn.
- Prepare data for machine learning.
- Train and compare multiple classification models.
- Evaluate model performance using metrics appropriate for an imbalanced classification problem.
- Explore how classification thresholds affect churn detection.
- Translate analytical findings into potential business recommendations.

## Dataset

**Source:** [IBM Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains 7,043 customer records and 21 columns, covering customer demographics, account information, subscribed services, billing details and churn status.

The target variable is `Churn`:

- `Yes`: The customer discontinued the service.
- `No`: The customer remained with the company.

The dataset includes information such as:

- Customer tenure
- Monthly and total charges
- Contract type
- Internet service
- Payment method
- Online security and technical support
- Customer demographics

## Tools and Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Google Colab

## Project Workflow

### 1. Data Understanding and Cleaning

- Examined dataset dimensions, column types and distributions.
- Checked for duplicate records and customer identifiers.
- Investigated missing values.
- Converted `TotalCharges` into a numerical variable.
- Handled blank `TotalCharges` values associated with customers with zero tenure.

### 2. Exploratory Data Analysis

Explored customer churn across categorical and numerical variables.

The analysis included:

- Overall churn distribution.
- Churn rates across contract types, payment methods and subscribed services.
- Customer tenure comparisons between churned and retained customers.
- Monthly and total charges by churn status.
- Correlations between numerical variables.

### 3. Data Preprocessing

Prepared the data for machine learning by:

- Removing the customer identifier from the feature set.
- Encoding the target variable into binary values.
- Splitting the dataset into training and testing sets using stratification.
- Imputing missing values.
- Scaling numerical variables.
- Applying one-hot encoding to categorical variables.

Preprocessing was implemented using Scikit-learn pipelines to keep transformations consistent and avoid fitting preprocessing steps on the test data.

### 4. Model Development

Four classification approaches were evaluated:

| Model | Purpose |
|---|---|
| Dummy Classifier | Establishes a baseline |
| Logistic Regression | Provides an interpretable classification baseline |
| Random Forest | Captures nonlinear relationships and feature interactions |
| Gradient Boosting | Uses sequential ensembles to improve predictive performance |

### 5. Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Recall and F1-score received particular attention because identifying customers who are likely to leave is important for customer-retention strategies.

## Model Results

The following results were obtained on the held-out test set using the default classification threshold of 0.50.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2654 |
| Logistic Regression | 0.7381 | 0.5043 | **0.7834** | **0.6136** | 0.8415 | 0.6325 |
| Random Forest | 0.7857 | 0.6286 | 0.4706 | 0.5382 | 0.8218 | 0.6085 |
| Gradient Boosting | **0.8048** | **0.6701** | 0.5214 | 0.5865 | **0.8454** | **0.6640** |

### Key Observations

- Gradient Boosting achieved the highest accuracy, ROC-AUC and PR-AUC.
- Logistic Regression achieved the highest recall and F1-score at the default threshold.
- Random Forest achieved higher precision than Logistic Regression but had lower recall.
- The Dummy Classifier achieved 73.46% accuracy while failing to identify churned customers, demonstrating why accuracy alone is insufficient for evaluating this problem.

### Classification Threshold Analysis

The Logistic Regression model was also evaluated at different classification thresholds.

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.30 | 0.429 | 0.928 | 0.587 |
| 0.35 | 0.447 | 0.904 | 0.598 |
| 0.40 | 0.467 | 0.866 | 0.607 |
| 0.45 | 0.483 | 0.840 | 0.613 |
| 0.50 | 0.504 | 0.783 | 0.614 |
| 0.55 | 0.525 | 0.751 | **0.618** |
| 0.60 | 0.540 | 0.706 | 0.612 |

The threshold of 0.55 produced the highest F1-score among the tested thresholds. Lower thresholds increased recall but also resulted in more false-positive predictions.

These threshold results are exploratory because the same test set was used to compare thresholds. A separate validation set should be used to select the threshold before an independent final test evaluation.

## Model Selection

Logistic Regression was selected as the provisional model for further development because it achieved the strongest recall and F1-score at the default threshold.

The choice reflects a business objective of identifying a larger proportion of customers who may leave, rather than maximizing accuracy alone.

Gradient Boosting remains a strong alternative, particularly where ranking and overall discrimination performance are priorities.

## Business Relevance

A churn prediction model can support customer-retention teams by helping them prioritize customers who may be at risk of discontinuing their subscriptions.

Potential applications include:

- Prioritizing high-risk customers for retention campaigns.
- Reviewing customer experiences associated with higher churn.
- Developing targeted retention strategies.
- Investigating whether contract options and service packages influence customer retention.
- Allocating customer-support resources more effectively.

Specific recommendations will be informed by the final feature-importance analysis.

## Repository Structure

```text
customer-churn-analysis/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   └── 03_churn_prediction_modeling.ipynb
│
├── models/
│   ├── logistic_regression_churn_pipeline.pkl
│   └── churn_threshold.pkl
│
├── images/
│   └── (project visualizations)
│
├── requirements.txt
├── .gitignore
└── README.md
```

The original dataset is not included in the repository. Refer to the Kaggle source above to obtain it.

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/customer-churn-analysis.git
```

2. Navigate into the project:

```bash
cd customer-churn-analysis
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the dataset from Kaggle and place the CSV in the appropriate data directory.

5. Open the notebooks in Google Colab or Jupyter Notebook and run them in numerical order.

## Future Improvements

- Perform hyperparameter tuning using cross-validation.
- Use a separate validation set for classification threshold selection.
- Evaluate feature importance and explain model predictions.
- Explore customer segmentation and churn-risk groups.
- Assess model performance using additional business-oriented evaluation criteria.
- Develop a simple interface for predicting churn risk for new customers.

## Author

**Data Science Portfolio Project**

Developed as a practical project in exploratory data analysis, predictive modelling and machine learning evaluation.