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


filename = 'data.xlsx'



df_deliver_from = pd.read_excel(filename, sheet_name='from')  # Shape (32776, 14)
df_deliver_to = pd.read_excel(filename, sheet_name='to') # Shape (657, 14)

df_delivery = df_deliver_from.merge(df_deliver_to, how='inner', left_on='Account', right_on='Account') # Shape (40619, 27)

train_set, test_set = train_test_split(df_delivery, test_size=0.2, random_state=42)

df_copy = train_set.copy()