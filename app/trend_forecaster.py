from prophet import Prophet
import pandas as pd

class TrendForecaster:
    def __init__(self):
        self.model = Prophet()

    def prepare_data(self, df, date_col='date'):
        df = df.copy()
        df['ds'] = pd.to_datetime(df[date_col])
        df['y'] = 1
        return df.groupby('ds').count().reset_index()[['ds', 'y']]

    def fit_and_predict(self, ts_df, periods=7):
        self.model.fit(ts_df)
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
