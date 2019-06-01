import model_main as m
# Front end packages
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.externals import joblib
import plotly.graph_objs as go

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




app = dash.Dash()

training_data = m.df_ML_test.Miles
training_labels = m.target

app.layout = html.Div(children=[
    html.H1(children='Simple Linear Regression', style={'textAlign': 'center'}),

    html.Div(children=[
        html.Label('Miles: '),
        dcc.Input(id='Miles', placeholder='Miles', type='text'),
        html.Div(id='result')
    ], style={'textAlign': 'center'}),

    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [
                go.Scatter(
                    x=training_data,
                    y=training_labels,
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                )
            ],
            'layout': go.Layout(
                xaxis={'type': 'log', 'title': 'Miles'},
                yaxis={'title': 'Plan Vs Actual'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                hovermode='closest'
            )
        }
    )
])



@app.callback(
    Output(component_id='result', component_property='children'),
    [Input(component_id='Miles', component_property='value')])

def model_input(Miles):
    if Miles is not None and Miles is not '':
        try:
            prediction = m.lin_Reg.predict(np.array(float(Miles)))
            return f'At {Miles} miles you will be {prediction} many hours off'
                
        except ValueError:
            return 'Unable to give Plan-Vs-Actual'


if __name__ == '__main__':
    app.run_server(debug=True)


