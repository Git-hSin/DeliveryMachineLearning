# Data analysis packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Front end packages
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.externals import joblib
import plotly.graph_objs as go


def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df





df_delivery = df_deliver_from.merge(df_deliver_to, how='inner', left_on='Account', right_on='Account') # Shape (40619, 27)



numeric_variables= ['PlanArrival', 'ActualArrival', 'Miles', 'PlanVsActual', 'TravelTime', 'HasNAs', 'lat', 'lng']
categorical_variables = ['Date', 'Day', 'Week', 'Month', 'Quarter', 'Driver', 'VehicleID',
       'Account',  'Supervisor', 'Shift',
       'ShiftHour', 'DepartureDoor', 'AccountName', 'Address',
       'City', 'State', 'ZipCode']


#df_ML = one_hot(df_delivery, categorical_variables)

#train_set, test_set = train_test_split(df_ML, test_size=0.2, random_state=42)

#df_copy = train_set.copy()