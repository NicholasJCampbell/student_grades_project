import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

class cleaning():
    numeric_columns = ['goout','failures','traveltime','Fedu','Medu','age','G1','G2']
    ordinal_columns = ['Mjob']
    categorical_columns = ['sex','address','paid','higher','romantic']
    
    class NumericDataTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, drop_grades=True):
            self.drop_grades = drop_grades
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            if self.drop_grades:
                X=X.drop(['G1','G2'], axis=1)
                return X
            else:
                return X
            
    numeric_pipeline_with_grades = make_pipeline (
        SimpleImputer(strategy="median").set_output(transform="pandas"),
        NumericDataTransformer(drop_grades=False),
        StandardScaler())
    
    numeric_pipeline_without_grades = make_pipeline (
        SimpleImputer(strategy="median").set_output(transform="pandas"),
        NumericDataTransformer(),
        StandardScaler())
    
    categorical_data_pipeline = make_pipeline (
        OneHotEncoder())
    
    ordinal_data_pipeline = make_pipeline (
        OrdinalEncoder())
    
    column_transformer_with_grades = ColumnTransformer ([
        ('num', numeric_pipeline_with_grades, numeric_columns),
        ('cat', categorical_data_pipeline, categorical_columns),
        ('ord', ordinal_data_pipeline, ordinal_columns)
    ])

    column_transformer_without_grades = ColumnTransformer ([
        ('num', numeric_pipeline_without_grades, numeric_columns),
        ('cat', categorical_data_pipeline, categorical_columns),
        ('ord', ordinal_data_pipeline, ordinal_columns)
    ])
