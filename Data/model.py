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

# Import ETL data

filename = 'data_fake_lat-lng.xlsx'

df_deliver_from = pd.read_excel(filename, sheet_name='from')  # Shape (32776, 14)
df_deliver_to = pd.read_excel(filename, sheet_name='to')

df_deliver_to_mapped = df_deliver_to[df_deliver_to.lat.isna() != True]

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



df_delivery = df_deliver_from.merge(df_deliver_to_mapped, how='inner', left_on='Account', right_on='Account') # Shape (40619, 27)

df_delivery = df_delivery.drop(['VehicleID', 'PlanArrival', 'ActualArrival', 'DepartureDoor', 'AccountName', 'Account', 'Address', 'City', 'State', 'ZipCode', 'FullAddress', 'HasNAs', 'ShiftHour'], axis=1)


numeric_variables= ['Miles', 'PlanVsActual', 'TravelTime',  'lat', 'lng']
categorical_variables = ['Date', 'Day', 'Week', 'Month', 'Quarter', 'Driver', 'Supervisor', 'Shift']


df_ML = one_hot(df_delivery, categorical_variables)

train_set, test_set = train_test_split(df_ML, test_size=0.2, random_state=42)

df_copy = train_set.copy()