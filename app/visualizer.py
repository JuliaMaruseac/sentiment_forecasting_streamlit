import plotly.express as px
import pandas as pd

def plot_sentiment_distribution(df):
    df = df[['negative', 'neutral', 'positive']]
    return px.bar(df.mean(), title="Средняя тональность", labels={"value": "Уровень", "index": "Класс"})

def plot_forecast(forecast):
    fig = px.line(forecast, x='ds', y='yhat', title='Прогноз интереса к теме')
    fig.add_traces([
        px.line(forecast, x='ds', y='yhat_upper').data[0],
        px.line(forecast, x='ds', y='yhat_lower').data[0]
    ])
    return fig
