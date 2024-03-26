#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 09:02:47 2024

@author: srilu
"""

# import libraries for data manipulation
import pandas as pd
import numpy as np

# import libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Importing libraries for building linear regression model
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split

# Importing libraries for scaling the data
from sklearn.preprocessing import MinMaxScaler

# Fitting linear model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

# Read the data
df = pd.read_csv('/Users/srilu/Documents/Financial Modelling/Case Study/Case 2 Sales Reps Salaries_Regression Analysis/Case 7.5 Sales Reps Salaries Dataset.csv')
df.head()

# Overview of the dataset shape and datatypes
df.shape
df.info()

# Checking for missing values in the dataset
df.isnull().sum()

# Statistical Summary of the data
round(df['Certficates'].describe(),2)

# Correlation Analysis
sns.heatmap(data=df[['Age','Years','Certficates','Feedback', 'Salary', 'NPS']].corr(), annot=True, fmt=".2f", cmap='Greys')
plt.xticks(rotation=45)

# Dropping the variable Sales_Rep, as it is an ID variable which is not useful for pediction
df.drop(columns = ['Sales_Rep'], inplace=True)

# Creating dummy variables for the categorical variables
df = pd.get_dummies(df, drop_first=True)

# Modeling
# We are removing the outcome variable from the feature set
features = df.drop(['Salary'], axis=1)
target = df['Salary']

# Spliting the data into training and testing subsets 
train_features, test_features, train_target, test_target = train_test_split(features,target,
                                                                            test_size =.2,random_state=1234)

# Creating an instance of the MinMaxScaler
scaler = MinMaxScaler()

# Applying fit_transform on the training features data
train_features_scaled = scaler.fit_transform(train_features)

# The above scaler returns the data in array format, below we are converting it back to pandas dataframe
train_features_scaled = pd.DataFrame(train_features_scaled,index = train_features.index, columns=train_features.columns)
train_features_scaled.head()

# Adding the intercept term
train_features_scaled = sm.add_constant(train_features_scaled)

# Calling the OLS algorithm on the train features and the target variable
ols_model_0 = sm.OLS(train_target, train_features_scaled)

# Fitting the Model
ols_res_0 = ols_model_0.fit()
print(ols_res_0.summary())

# Compute VIF values for each feature
vif_series = pd.Series(
    [variance_inflation_factor(train_features_scaled.values, i) for i in range(train_features_scaled.shape[1])],
    index=train_features_scaled.columns,
    dtype=float
)

print("VIF Scores: \n\n{}\n".format(vif_series))

# Let's build the model again
train_features_scaled_new = train_features_scaled.drop(['Personality_Sentinel'], axis=1)

ols_model_1 = sm.OLS(train_target, train_features_scaled_new)

ols_res_1 = ols_model_1.fit()

print(ols_res_1.summary())

vif_series = pd.Series(
    [variance_inflation_factor(train_features_scaled_new.values, i) for i in range(train_features_scaled_new.shape[1])],
    index=train_features_scaled_new.columns,
    dtype=float
)

print("VIF Scores: \n\n{}\n".format(vif_series))

# Checking for the assumptions and rebuilding the model
# Mean of residuals should be 0 and normality of error terms
residual = ols_res_1.resid
residual.mean()

# Plot histogram of residuals
sns.histplot(residual, kde = True, color='darkgrey')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

# Linearity of the variables

# Predicted values
fitted = ols_res_1.fittedvalues

sns.residplot(x=fitted, y=residual, color="darkgrey")
plt.xlabel("Fitted Values")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()

# Log transformation on the target variable
train_target_log = np.log(train_target)

# Fitting new model with the transformed target variable
ols_model_1 = sm.OLS(train_target_log, train_features_scaled_new)

ols_res_1 = ols_model_1.fit()

# Predicted values
fitted = ols_res_1.fittedvalues
residual1 = ols_res_1.resid

sns.residplot(x=fitted, y=residual1, color="darkgrey")
plt.xlabel("Fitted Values")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()

print(ols_res_1.summary())

from statsmodels.stats.diagnostic import het_white
from statsmodels.compat import lzip
import statsmodels.stats.api as sms

name = ["F statistic", "p-value"]
test = sms.het_goldfeldquandt(train_target_log, train_features_scaled_new)
lzip(name, test)

# Let's make the final test predictions

without_const = train_features_scaled.iloc[:, 1:]

test_features = test_features[list(without_const)]

# Applying transform on the test data
test_features_scaled = scaler.transform(test_features)

test_features_scaled = pd.DataFrame(test_features_scaled, columns = without_const.columns)

test_features_scaled = sm.add_constant(test_features_scaled)

test_features_scaled = test_features_scaled.drop(['Personality_Sentinel'], axis=1)

test_features_scaled.head()

# R-squared
print(ols_res_1.rsquared)

# Mean Squared Error
print(ols_res_1.mse_resid)

# Root Mean Squared Error
print(np.sqrt(ols_res_1.mse_resid))

# Cross Validation Scores
linearregression = LinearRegression()
cv_Score11 = cross_val_score(linearregression, train_features_scaled_new, train_target_log, scoring='r2')
cv_Score12 = cross_val_score(linearregression, train_features_scaled_new, train_target_log, scoring='neg_mean_squared_error')

print("RSquared: %0.3f (+/- %0.3f)" % (cv_Score11.mean(), cv_Score11.std()*2))
print("Mean Squared Error: %0.3f (+/- %0.3f)" % (-1*cv_Score12.mean(), cv_Score12.std()*2))

# Predictions on the Test Dataset

# These test predictions will be on a log scale
test_predictions = ols_res_1.predict(test_features_scaled)

# We are converting the log scale predictions to its original scale
test_predictions_inverse_transformed = np.exp(test_predictions)

# Calculate R-squared of the test data
test_r2_score = r2_score(test_target, test_predictions_inverse_transformed)

