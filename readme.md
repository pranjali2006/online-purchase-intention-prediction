# Online Purchase Intention Prediction using Logistic Regression

## 1. Project Overview

The goal of this project is to predict whether an online shopping visitor will complete a purchase or not based on their browsing behaviour and session information.

This project uses **Logistic Regression**, a machine learning classification algorithm, to predict purchase intention.

The main aim is to understand customer behaviour, perform Exploratory Data Analysis (EDA), prepare the data, train a classification model, and evaluate its performance.

## 2. Dataset

**Dataset:** Online Shoppers Purchasing Intention Dataset

**Source:** UCI Machine Learning Repository (also available on Kaggle)

The dataset contains:
- 12,330 online shopping sessions
- 18 columns (17 input features and 1 target column)

The target column is `purchase`:
- `True` – The visitor completed a purchase.
- `False` – The visitor did not complete a purchase.

## 3. Project Workflow

### Step 1: Importing Libraries and Loading the Dataset

Imported the required Python libraries, including:
- Pandas and NumPy for data handling.
- Matplotlib and Seaborn for data visualization.
- Scikit-learn for preprocessing, model training, and evaluation.

Loaded the dataset into a Pandas DataFrame.

### Step 2: Understanding the Data

Used the following functions to understand the dataset:

- `head()` – Viewed the first few rows.
- `tail()` – Viewed the last few rows.
- `info()` – Checked column names, data types, and non-null values.
- `describe()` – Studied the statistical summary of numerical columns.

Also renamed the columns to make them easier to understand and work with.

### Step 3: Data Quality Checking

Checked the dataset for:

- Missing values using `isnull()`.
- Duplicate records using `duplicated()`.
- Potential outliers using boxplots and statistical summaries.

This step helped understand the quality and reliability of the data before performing EDA.

### Step 4: Exploratory Data Analysis (EDA)

Performed EDA in three stages.

**A. Univariate Analysis**

Studied individual features to understand their distributions.

First, divided the features into:
- Numerical features
- Categorical features
- Binary features

Used histograms, boxplots, and count plots to understand the data.

**B. Bivariate Analysis**

Studied the relationship between individual features and the target column (`purchase`).

Used:
- Boxplots for numerical features against purchase.
- Bar plots to compare purchase rates across categorical features.

**C. Multivariate Analysis**

Used a correlation heatmap to understand relationships between numerical features and identify highly correlated feature pairs.

### Step 5: EDA Conclusions and Feature Understanding

Based on the EDA, identified important patterns and relationships in the data.

Studied the correlation of numerical features with the target variable and examined possible feature redundancy.

Also identified `page_value` as a potential data leakage feature because it may contain information that is not available at the time of prediction.

Therefore, `page_value` was excluded from model training.

### Step 6: Separating Features and Target

Separated the dataset into:

- `X` – Input features used for prediction.
- `y` – Target column (`purchase`).

### Step 7: Train-Test Split

Split the data into training and testing sets using `train_test_split()`.

Used **stratified splitting** to maintain approximately the same purchase and non-purchase proportions in both sets.

This was important because the dataset contains an imbalanced target variable.

### Step 8: Data Preprocessing

Applied different preprocessing techniques to numerical and categorical features.

- **StandardScaler:** Used to scale numerical features.
- **OneHotEncoder:** Used to convert categorical features into numerical form.

### Step 9: Building a Machine Learning Pipeline

Created a Scikit-learn Pipeline using:

- `ColumnTransformer` for preprocessing.
- `StandardScaler` for numerical features.
- `OneHotEncoder` for categorical features.
- `LogisticRegression` as the classification algorithm.

Using a Pipeline ensured that preprocessing was fitted only on training data and then applied to validation and test data, helping prevent data leakage.

### Step 10: Model Training

Trained the Logistic Regression model using the training data.

Generated predictions and purchase probabilities to evaluate the model.

### Step 11: Model Evaluation

Evaluated the model using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

The initial model achieved approximately 84% accuracy at the default threshold of 0.5.

However, the confusion matrix showed that the model was missing many actual purchases.

This highlighted the limitation of using accuracy alone for an imbalanced classification problem.

### Step 12: Threshold Comparison and Selection

Tested different classification thresholds to understand the trade-off between precision and recall.

Compared thresholds such as:

`0.1, 0.2, 0.3, 0.4, 0.5`

Observed that lowering the threshold increased recall but also increased false-positive predictions.

Based on the initial comparison, a threshold around 0.2 provided a better balance between precision and recall.

The final selected threshold for this project was **0.21**, based on the threshold-selection process.

### Step 13: Final Model Evaluation

Used the selected threshold to convert predicted purchase probabilities into final class predictions.

Evaluated the model on unseen test data using the confusion matrix and classification report.

This helped understand how well the model generalizes to data that was not used during training.

## 4. Technologies and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 5. Key Learnings

Through this project, I learned how to:

- Understand and clean a real-world dataset.
- Perform univariate, bivariate, and multivariate EDA.
- Identify relationships between features and the target variable.
- Understand class imbalance and its effect on model evaluation.
- Apply feature scaling and one-hot encoding.
- Build a machine learning pipeline.
- Understand and avoid potential data leakage.
- Train and evaluate a Logistic Regression model.
- Compare classification thresholds using precision, recall, and F1-score.

## 6. Conclusion

This project helped me understand the complete machine learning workflow, from loading and exploring data to preprocessing, model training, and evaluation.

It also helped me understand why accuracy alone is not enough for imbalanced classification problems and how threshold selection can improve the model's ability to identify potential purchasers.