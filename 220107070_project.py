# -*- coding: utf-8 -*-
"""Project_final2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14ZzReTfxCLeO3hHBcWVtbWHc38sN4-aY

In this section, we import essential libraries for data manipulation, visualization, preprocessing, model building, and evaluation. We also suppress warnings for a cleaner output.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb

import warnings

warnings.filterwarnings('ignore')



"""This cell loads the Air Quality dataset from a CSV file, inspects its shape and column names, and displays the first few rows for an initial look at the data."""

df = pd.read_csv("AirQualityUCI.csv", sep=';', decimal=',')
print(df.shape)
print(df.columns)
df.head(100)

# df.fillna(-200, inplace=True)

"""In this step, we:

Drop the last two empty columns from the dataset.

Fill missing values in numeric columns with their respective column means.

Convert Date and Time columns into a single Datetime column using a specific format.

Strip any whitespace from column names and drop the original Date and Time columns.
"""

print("Before cleaning:", df.shape)
df = df.iloc[:, :-2]

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].applymap(lambda x: np.nan if x < 0 else x)

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

print("After cleaning:", df.shape)

df['Datetime'] = pd.to_datetime(
    df['Date'].str.strip() + ' ' + df['Time'].str.strip(),
    format='%d/%m/%Y %H.%M.%S'
)

df.columns = df.columns.str.strip()
df.drop(columns=['Date', 'Time'], inplace=True)

print("After cleaning:", df.shape)
df.head()

df.dropna(inplace=True)
print(f"After dropping rows with NaN values, the DataFrame shape is: {df.shape}")





nan_counts = df.isna().sum()
total_nan = nan_counts.sum()

print(f"NaN counts per column:\n{nan_counts}")
print(f"\nTotal NaN values in the DataFrame: {total_nan}")

df.info()
df.describe()

"""Plotting Histograms: It visualizes the distribution of all numeric columns in the dataset using histograms, with Kernel Density Estimation (KDE) overlaid for a smoother visualization.

Removing Outliers: It uses the Interquartile Range (IQR) method to remove rows that contain outliers. Any data point that lies outside 1.5 times the IQR above the third quartile or below the first quartile is considered an outlier.
"""

for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()

Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

df_cleaned = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

print(f"Data shape after removing outliers: {df_cleaned.shape}")

print("RH" in df.columns)  # should be True

df=df_cleaned

"""This step plots the distribution of the Relative Humidity (RH) column using a histogram with a kernel density estimate (KDE) to understand its distribution.


"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7, 4))
sns.histplot(df['RH'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Relative Humidity')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Count')
plt.grid(True)
plt.show()

"""In this step, we calculate and visualize the correlation matrix of the dataset, excluding the Datetime column. The heatmap helps identify relationships between the features."""

plt.figure(figsize=(10, 8))
corr_matrix = df.drop(columns=['Datetime']).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

"""This step creates a line plot to visualize the change in relative humidity (RH) over time, using the Datetime column on the x-axis.


"""

plt.figure(figsize=(12, 4))
sns.lineplot(x='Datetime', y='RH', data=df, color='green')
plt.title('Relative Humidity Over Time')
plt.xlabel('Time')
plt.ylabel('Relative Humidity (%)')
plt.tight_layout()
plt.show()

"""In this step, we extract time-based features like Hour, Month, Day, and Weekday from the Datetime column. We then plot a boxplot to visualize the variation of relative humidity (RH) throughout the day, based on the hour.


"""

df['Hour'] = df['Datetime'].dt.hour
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day
df['Weekday'] = df['Datetime'].dt.dayofweek

plt.figure(figsize=(7, 4))
sns.boxplot(x='Hour', y='RH', data=df)
plt.title('Hourly Variation in Relative Humidity')
plt.xlabel('Hour of Day')
plt.ylabel('Relative Humidity (%)')
plt.grid(True)
plt.show()

"""This step creates a scatter plot to visualize the relationship between temperature (T) and relative humidity (RH), with axis limits set for better clarity."""

plt.figure(figsize=(6, 4))
sns.scatterplot(x='T', y='RH', data=df)
plt.xlim(0, 50)  # Set the x-axis limit to start from 0
plt.ylim(0, 100)
plt.title('Temperature vs Relative Humidity')
plt.xlabel('Temperature (°C)')
plt.ylabel('Relative Humidity (%)')
plt.grid(True)
plt.show()

"""In this step, we drop the Datetime column from the dataset as it's not useful for modeling, and we inspect the first few rows of the modified dataset."""

df_model = df.drop(columns=['Datetime'])
df_model.head()

"""In this step, we separate the features (X) and the target variable (y). The features include all columns except RH, which is the target variable (relative humidity)."""

X = df_model.drop(columns=['RH'])
y = df_model['RH']

"""In this step, we split the data into training and testing sets, with 80% of the data used for training and 20% for testing. We also set a random seed to ensure reproducibility."""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""In this step, we apply feature scaling to the training and testing data. While tree-based models don't require scaling, models like KNN, SVM, and linear regression benefit from it. We use StandardScaler to standardize the features."""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""In this step, we define a function to evaluate the performance of a model. The function fits the model, makes predictions, and calculates common regression metrics: R², Mean Absolute Error (MAE), and Mean Squared Error (MSE)."""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model: {model.__class__.__name__}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print("-" * 50)

    return model, r2, mae, mse

"""In this step, we define a list of machine learning models and evaluate their performance using the evaluate_model function. The evaluation results (R², MAE, MSE) are stored in a list for further analysis."""

models = [
    LinearRegression(),
    Ridge(),
    ElasticNet(),
    RandomForestRegressor(n_estimators=100, random_state=42),
    DecisionTreeRegressor(random_state=42),
    KNeighborsRegressor(),
    GradientBoostingRegressor(n_estimators=100, random_state=42),
    SVR(),
    xgb.XGBRegressor(n_estimators=100, random_state=42),
    lgb.LGBMRegressor(n_estimators=100, random_state=42)
]

results = []

for model in models:
    model, r2, mae, mse = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    results.append([model.__class__.__name__, r2, mae, mse])

"""In this step, we convert the evaluation results into a DataFrame and sort the models by their R² scores. We then create a bar plot to compare the R² scores of different models visually."""

results_df = pd.DataFrame(results, columns=['Model', 'R²', 'MAE', 'MSE'])
results_df = results_df.sort_values(by='R²', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='R²', y='Model', data=results_df, palette='viridis')
plt.title('Model Comparison - R² Scores')
plt.xlabel('R²')
plt.ylabel('Model')
plt.show()

"""R² is also called the coefficient of determination.

It measures the proportion of variance in the dependent variable that is explained by the independent variables in the model.

The value of R² ranges from 0 to 1:

0 means the model explains none of the variance (very poor fit).

1 means the model explains all the variance (perfect fit).

Values between 0 and 1 indicate partial explanation of the variance.

How to Interpret R² in This Plot
Each bar represents a regression model and its corresponding R² score.

Higher R² means the model predicts the target variable more accurately.

For example, in this plot, models like RandomForestRegressor and GradientBoostingRegressor have higher R² scores, indicating they fit the data better than models like ElasticNet or KNeighborsRegressor.

In this step, we define a function that not only evaluates the model's performance with common metrics (R², MAE, MSE) but also calculates the variance reduction (an approximation of R²) based on the model's predictions.
"""

def evaluate_model_with_variance_reduction(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    total_variance = y_test.var()
    explained_variance = 1 - (mse / total_variance)

    print(f"Model: {model.__class__.__name__}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Variance Reduction: {explained_variance:.4f}")
    print("-" * 50)

    return model, r2, mae, mse, explained_variance

"""This step defines a function to extract and visualize the feature importance for tree-based models. The function creates a bar plot and returns a DataFrame with the features sorted by their importance."""

def extract_feature_importance(model, X_train):
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("Feature Importance:")
    print(importance_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

    return importance_df



"""In this step, we define and evaluate several models (Random Forest, Linear Regression, SVR, and Decision Tree) based on their performance using metrics like R², MAE, MSE, and Variance Reduction. Additionally, we extract and visualize feature importance for tree-based models."""

results = []

for model in models:
    print(f"Evaluating {model.__class__.__name__}...")

    model, r2, mae, mse, variance_reduction = evaluate_model_with_variance_reduction(
        model, X_train_scaled, X_test_scaled, y_train, y_test
    )

    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = extract_feature_importance(model, X_train)

    results.append([model.__class__.__name__, r2, mae, mse, variance_reduction, feature_importance])

results_df = pd.DataFrame(results, columns=['Model', 'R²', 'MAE', 'MSE', 'Variance Reduction', 'Feature Importance'])
results_df = results_df.sort_values(by='R²', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Variance Reduction', y='Model', data=results_df, palette='viridis')
plt.title('Model Comparison - Variance Reduction')
plt.xlabel('Variance Reduction')
plt.ylabel('Model')
plt.show()

"""Summary of Purpose
The overall goal of this process is to evaluate and compare the performance of multiple regression models, not only in terms of common metrics like R², MAE, and MSE, but also in terms of variance reduction and feature importance (for tree-based models).

This helps in selecting the best model based on a variety of performance indicators, and also provides insights into which features are most important for predicting the target variable.

This approach enables you to choose the most effective model for predicting relative humidity, backed by both quantitative performance metrics and feature analysis.

In this step, the best performing model, RandomForestRegressor, is selected based on the previous evaluation results. The chosen model is then trained using the scaled training data to make predictions on the test set.
"""

best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train_scaled, y_train)

"""In this step, we perform hyperparameter tuning on the RandomForestRegressor using GridSearchCV. The hyperparameters being tuned include the number of estimators (n_estimators), maximum depth (max_depth), and minimum samples required to split an internal node (min_samples_split). The grid search is performed using 3-fold cross-validation, and the model with the best R² score is selected."""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring='r2')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

"""his step visualizes the feature importances of the best RandomForestRegressor model after hyperparameter tuning. The plot shows the relative importance of each feature in predicting relative humidity (RH). Higher bars represent more influential features."""

import matplotlib.pyplot as plt
import seaborn as sns

importances = best_rf.feature_importances_
feature_names = X_train.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importances")
plt.show()

"""In this step, the trained model (best_model) is used to make predictions on the test data (X_test_scaled) for relative humidity (RH). The predictions are stored in y_pred_final."""

y_pred_final = best_model.predict(X_test_scaled)

"""This step evaluates the final model's performance using R², MAE, and MSE to assess its accuracy and prediction error on the test data."""

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_test, y_pred_final)
mae = mean_absolute_error(y_test, y_pred_final)
mse = mean_squared_error(y_test, y_pred_final)

print(f"Final Model Evaluation:")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")

"""In this step, we visualize the predicted relative humidity values against the actual values from the test set to assess the model's performance."""

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_final, color='blue', alpha=0.6)
plt.xlim(0), plt.ylim(0)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Predicted vs Actual Relative Humidity')
plt.xlabel('Actual Relative Humidity')
plt.ylabel('Predicted Relative Humidity')
plt.show()



"""## Model Evaluation and Residuals Distribution  
We evaluate the performance of the RandomForestRegressor model by calculating R², MAE, and MSE. Additionally, we plot the distribution of residuals (the differences between the actual and predicted values) to check the model's error behavior.
"""

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

preds = best_rf.predict(X_test)
print("R²:", r2_score(y_test, preds))
print("MAE:", mean_absolute_error(y_test, preds))
print("MSE:", mean_squared_error(y_test, preds))

sns.histplot(y_test - preds, kde=True)
plt.title("Residuals Distribution")
plt.show()

"""## Model Saving and Loading  
The trained model is saved to a file using `joblib` for future use, and it is then loaded back to confirm that the saved model can be successfully reloaded for predictions.
"""

import joblib

joblib.dump(best_model, 'best_model_final.pkl')


loaded_model = joblib.load('best_model_final.pkl')

# !pip install streamlit
# !pip install pyngrok

# %%writefile app.py
# import streamlit as st
# import joblib
# import numpy as np

# # Load the saved model
# model = joblib.load('best_model_final.pkl')

# # Streamlit app UI
# st.title('Relative Humidity Prediction')
# st.write('Enter the feature values below to predict the relative humidity.')

# # Input fields for model features
# feature_1 = st.number_input('Feature 1')
# feature_2 = st.number_input('Feature 2')
# feature_3 = st.number_input('Feature 3')

# # Predict button
# if st.button('Predict'):
#     input_data = np.array([feature_1, feature_2, feature_3]).reshape(1, -1)
#     prediction = model.predict(input_data)
#     st.write(f'Predicted Relative Humidity: {prediction[0]:.2f}')

# from pyngrok import ngrok
# ngrok.set_auth_token("2wDy0wcSRsgy7F2DnRxkJNrNogB_2a6yQvNoaRTdz8SfwxCEy")

# public_url = ngrok.connect(addr='8501')
# print(f"Streamlit app is live at {public_url}")

# sample_input = np.array([[2.5, 1360.0, 150.0, 11.2, 1049.0, 166.0,
#                           1185.0, 113.0, 948.0, 774.0, 25.0, 0.85,
#                           14, 6, 15, 2]])

