# Data analysis packages
from sklearn import tree
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Import ETL data

filename = 'data_fake_lat-lng.xlsx'

df_deliver_from = pd.read_excel(filename, sheet_name='from')  # Shape (32776, 14)
df_deliver_to = pd.read_excel(filename, sheet_name='to')

df_deliver_from_mapped = df_deliver_from.dropna()
df_deliver_to_mapped = df_deliver_to.dropna()

import sqlite3 as db
conn = db.connect('data.db') 
df_delivery = pd.read_sql_query("""SELECT * FROM delivery""",con=conn)
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



#df_delivery = df_deliver_from_mapped.merge(df_deliver_to_mapped, how='inner', left_on='Account', right_on='Account') # Shape (40619, 27)

df_delivery = df_delivery.drop(['VehicleID', 'PlanArrival', 'ActualArrival', 'DepartureDoor', 'AccountName', 'Account', 'Address', 'City', 'State', 'ZipCode', 'FullAddress', 'HasNAs', 'ShiftHour'], axis=1)

# Find non-ints
# df.applymap(lambda x: isinstance(x, (int, float)))


numeric_variables= ['Miles', 'PlanVsActual', 'TravelTime',  'lat', 'lng']
categorical_variables = ['Month','Dt','Year', 'Day', 'Week', 'Quarter', 'Driver', 'Supervisor', 'Shift']


df_ML = one_hot(df_delivery, categorical_variables)

df_ML_test = df_ML.drop(categorical_variables, axis=1)

target = df_ML_test['PlanVsActual']
target_names = ['late', 'on-time']

data = df_ML_test.drop('PlanVsActual', axis=1)
feature_names = data.columns

lin_reg = LinearRegression()
lin_Reg = lin_reg.fit(data, df_ML_test.PlanVsActual)


from sklearn import tree
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.25, random_state=42)
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)
clf.score(X_test, y_test)


