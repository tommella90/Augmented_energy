import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image
import plotly.express as px
import plotly.io as pio
import fastparquet
pio.renderers.default = "browser"

import warnings
warnings.filterwarnings("ignore")

class AugmentedEnergyPrediction:
    """
    Initializes the AugmentedEnergyPrediction class.

    Parameter:
    consumption_group (str): The group of consumption to use for the prediction. Can be one of \n-"BASE" \n-"HP" \n-"HC"
    _________________________________________________________
    """
    def __init__(self, consumption_group):
        self.consumption_group = consumption_group

    def load_data(self):
        #Loads data for the specified consumption group.
        #Returns:
        #pd.DataFrame: A DataFrame containing the data for the specified consumption group.
        if self.consumption_group=="BASE":
            dataframe = pd.read_parquet("data/RES11_BASE.parquet.gzip", engine="fastparquet")
        elif self.consumption_group=="HP":
            dataframe = pd.read_parquet("data/RES2_HP.parquet.gzip", engine="fastparquet")
        elif self.consumption_group=="HC":
            dataframe = pd.read_parquet("data/RES2_HC.parquet.gzip", engine="fastparquet")
        else:
            print("File name not found")

        return dataframe



    def load_past_and_future_data(self):
        #Loads past and future data for the specified consumption group.
        #Returns:
        #pd.DataFrame: A DataFrame containing the past and future data for the specified consumption group.

        past = self.load_data()
        past['isFuture'] = False

        # future data
        future_df = pd.read_parquet("data/temperature_forecast.parquet.gzip", engine="fastparquet")
        future_df.index = pd.date_range(start=past.index.max(), periods=len(future_df.index), freq='H')
        future_df = future_df.sort_index()
        future_df['isFuture'] = True

        # combine past and future data
        all_df = pd.concat([past, future_df], axis=0)
        all_df.index = pd.to_datetime(all_df.index, utc=True)
        all_df = all_df.dropna(how='any',subset=['temperature'])

        return all_df


    def create_features(self):
        #Creates features for the past and future data.
        #Returns:
        #pd.DataFrame: A DataFrame containing the past and future data with added features.

        dataframe = self.load_past_and_future_data()

        # create time-features
        dataframe['hours'] = dataframe.index.hour
        dataframe['days'] = dataframe.index.dayofweek
        dataframe['quarters'] = dataframe.index.quarter
        dataframe['months'] = dataframe.index.month
        dataframe['years'] = dataframe.index.year
        dataframe['dayofyear'] = dataframe.index.dayofyear

        # create lag features
        target_map = dataframe['coefficient_value'].to_dict()
        dataframe['lag_1year'] = (dataframe.index - pd.Timedelta('364 days')).map(target_map)  ## same day last year
        dataframe['lag_2year'] = (dataframe.index - pd.Timedelta('728 days')).map(target_map)  ## same day last year
        return dataframe


    def select_future_dataframe(self):
        #Select future dates.
        #Returns:
        #pd.DataFrame: A DataFrame containing the future data.
        past_and_future = self.create_features()
        past_and_future = past_and_future.loc[past_and_future['isFuture']==True]

        features = ['temperature', 'hours', 'days', 'quarters', 'months', 'years',
                    'dayofyear', 'lag_1year', 'lag_2year', 'coefficient_value']
        past_and_future = past_and_future[features]
        past_and_future = past_and_future.dropna(how='any', subset=['lag_1year', 'lag_2year'])

        # select features for the model
        features = ['temperature', 'hours', 'days', 'quarters', 'months', 'years',
                    'dayofyear', 'lag_1year', 'lag_2year']
        past_and_future = past_and_future[features]

        return past_and_future


    def load_model(self):
        if self.consumption_group=="BASE":
            with open("models/model_xgb_BASE.pickle", "rb") as f:
                model = pickle.load(f)

        elif self.consumption_group=="HP":
            with open("models/model_xgb_HP.pickle", "rb") as f:
                model = pickle.load(f)

        elif self.consumption_group=="HC":
            with open("models/model_xgb_HC.pickle", "rb") as f:
                model = pickle.load(f)

        return model


    def print_model(self):
        """
        Print the model parameters
        _________________________________________________________
        """
        print(self.load_model())


    def predict_year_energy_consumption(self, starting_date="2022-10-31"):
        """
        Predict energy consumption for the following year.

        :parameter:
        starting_date (str): The starting date for the prediction. format: YYYY-MM-DD

        :return:
        pd.DataFrame: A DataFrame containing hourly forecasted consumption and the date (as index).
        _________________________________________________________
        """
        dataframe = self.select_future_dataframe()
        model = self.load_model()

        time_range = pd.date_range(pd.to_datetime(starting_date),
                                   periods=len(dataframe.index),
                                   freq='1h')

        year_energy_prediction = pd.DataFrame(model.predict(dataframe))
        year_energy_prediction.index = time_range
        year_energy_prediction.columns = ["energy_consumption"]
        return year_energy_prediction


    def predict_day_energy_consumption(self, day):
        """
        Predict energy consumption for a selected day within a year.

        :parameter:
        starting_date (str): The starting date for the prediction. format: YYYY-MM-DD

        :return:
        pd.DataFrame: A DataFrame containing hourly forecasted energy consumption
        for the selected day
        _________________________________________________________
        """
        df = self.select_future_dataframe()
        day_to_predict = df[df.index.date.astype(str) == day]

        model = self.load_model()

        day_energy_prediction = pd.DataFrame(model.predict(day_to_predict))
        day_energy_prediction.columns = ['energy_consumption']
        return day_energy_prediction


    def plot_year_prediction(self):
        """ Plot the predicted energy consumption for the following year.

        :parameter:
        past (bool): Default=False. If True, plot both the past data and the forecasted ones
        _________________________________________________________
        """
        prediction = self.predict_year_energy_consumption()
        prediction['Time'] = "Future"

        past = self.load_data()
        past.rename(columns={'coefficient_value': 'energy_consumption'}, inplace=True, errors='raise')
        past['Time'] = "Past"
        past = past.drop(columns=['temperature', 'profile_class'])

        all_data = pd.concat([past, prediction], axis=0)
        all_data.index = pd.to_datetime(all_data.index)
        all_data = all_data.sample(n=6000, random_state=42)
        all_data['Color']= all_data['Time'].apply(lambda x: 'blue' if (x == "Past") else 'green')

        all_data.columns = ['Energy consumption', 'Time', 'Color']

        title = f"ENERGY CONSUMPTION FORECAST, 2022-10-31 - 2023-10-30. Consumption group"

        import seaborn as sns
        fig, ax = plt.subplots(figsize=(15,5))

        sns.scatterplot(all_data, x=all_data.index, y="Energy consumption")
        plt.title(title)
        fig.show()


    def plot_day_prediction(self, day):
        """ Plot the predicted energy consumption for the following year.

        :parameter:
        day (str): The day for the prediction. format: YYYY-MM-DD
        _________________________________________________________
        """
        prediction = self.predict_day_energy_consumption(day)
        prediction.reset_index(inplace=True)
        prediction.columns = ['Hours', 'Energy Consumption']

        title = f"DAILY CONSUMPTION FORECAST: {day}. Consumption group: {self.consumption_group}"
        fig = px.bar(prediction, x="Hours", y="Energy Consumption",
                     color="Energy Consumption",
                     height=500, width=700,
                     title=title,
                     labels=['Hours', 'Energy Consumption'],
                     color_continuous_scale=px.colors.sequential.Greens)

        fig.update_xaxes(tickmode = 'array',
                         tickvals = list(range(0,24)))
        fig.show()


"""
fig = px.scatter(all_data, x=all_data.index, y="Energy consumption",
                         color_discrete_sequence=["limegreen", "darkgreen"],
                         color="Time",
                         title=title)
        fig.update_traces(marker_size=6)
        fig.update_layout(
            xaxis_tickformat = '%Y'
        )

        fig.show()
"""

#%%
