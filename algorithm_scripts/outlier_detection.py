from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
#we use knn algorithm
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

def IQR(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    upper_bounds = Q3 + 1.5*(Q3-Q1)
    lower_bounds = Q1 - 1.5*(Q3-Q1)
    outliers = ((data > upper_bounds) | (data < lower_bounds))
    return outliers


def iqrImpute(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bounds = Q1 - 1.5 * IQR
    upper_bounds = Q3 + 1.5 * IQR
    
    columns = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6', 'Column7', 'Column8', 'Column9', 'Column14', 'Column15']
    for column in data.columns:
        if column in columns:
            data[column] = data[column].clip(lower=lower_bounds[column], upper=upper_bounds[column])
        
        
    
    
    return data
