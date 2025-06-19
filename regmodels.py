import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import scatter_matrix



def linreg_model(x):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_transformed_with_grades, y_train)
    lin_reg.fit(X_train_transformed_without_grades, y_train)
    return x

def lassoreg_model(x):
    lasso_reg = Lasso()
    lasso_params = {'alpha' : [0.05, 0.1, 0.3, 1, 3, 5],}
    lasso_grid = GridSearchCV(lasso_reg, lasso_params, cv=10, n_jobs=-1)
    lasso_grid_with = lasso_grid.fit(X_train_transformed_with_grades, y_train)
    lasso_grid_without = lasso_grid.fit(X_train_transformed_without_grades, y_train)
    return x

class svmreg_model():
    svm_reg = SVR()
    svm_reg.fit(X_train_transformed_with_grades, y_train)
    svm_reg.fit(X_train_transformed_without_grades, y_train)

class ridgereg_model():
    ridge_reg = Ridge()
    ridge_params = {'alpha' : [0.05, 0.1, 0.3, 1, 3, 5, 6, 8, 10, 15, 30, 50, 75]}
    ridge_grid = GridSearchCV(ridge_reg, ridge_params, cv=10, n_jobs=-1)
    ridge_grid_with = ridge_grid.fit(X_train_transformed_with_grades, y_train)
    ridge_grid_without = ridge_grid.fit(X_train_transformed_without_grades, y_train)

class randomforest_model():
    forest_reg = RandomForestRegressor(random_state=42)
    forest_params = {'n_estimators' : [100, 200, 300]}
    forest_grid = GridSearchCV(forest_reg, forest_params, cv=10, n_jobs=-1)
    forest_grid_with = forest_grid.fit(X_train_transformed_with_grades, y_train)
    forest_grid_without = forest_grid.fit(X_train_transformed_without_grades, y_train)