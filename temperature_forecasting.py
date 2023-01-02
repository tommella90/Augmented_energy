import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import warnings
warnings.filterwarnings("ignore")


#%%
actual = pd.read_csv("data/weather_actuals.csv.gz", compression='gzip')

#%%
stations = pd.read_csv("data/weather_stations.csv")
actual = actual[actual['station_id'].isin(list(stations['id'].unique()))]
temperature = pd.merge(actual, stations, left_on='station_id', right_on='id', how='left')
temperature = temperature.set_index('recorded_at')
temperature.index = pd.to_datetime(temperature.index)
temperature = temperature.sort_index()


#%%
def create_features(dataframe):
    dataframe['hours'] = dataframe.index.hour
    dataframe['days'] = dataframe.index.dayofweek
    dataframe['quarters'] = dataframe.index.quarter
    dataframe['months'] = dataframe.index.month
    dataframe['years'] = dataframe.index.year
    dataframe['dayofyear'] = dataframe.index.dayofyear

    target_map = dataframe['temperature'].to_dict()
    dataframe['lag_1year'] = (dataframe.index - pd.Timedelta('364 days')).map(target_map)  ## same day last year
    dataframe['lag_2year'] = (dataframe.index - pd.Timedelta('728 days')).map(target_map)  ## same day last year
    #dataframe['lag_3year'] = (dataframe.index - pd.Timedelta('1092 days')).map(target_map)  ## same day last year
    return dataframe


#%%
def create_future_dates(df, time_range='1 y', frequency='1h'):
    future_boundary = df.index.max() + pd.Timedelta(time_range)
    future = pd.date_range(df.index.max(), future_boundary, freq=frequency)
    future_df = pd.DataFrame(index=future)
    future_df['isFuture'] = True
    df['isFuture'] = False
    new_df = pd.concat([df, future_df])
    return new_df


#%%
def predict_temperature_by_station(station_name, dataframe):

    # select station and create features
    station = dataframe[dataframe['name']==station_name]
    station_features = create_features(station)

    # train the model on the past data
    features = ['hours', 'days', 'quarters', 'months', 'years', 'dayofyear',
                'lag_1year', 'lag_2year']
    target = 'temperature'

    x_all = station_features[features]
    y_all = station_features[target]

# create and fit the model
    xgb_reg = xgb.XGBRegressor(base_score=0.5,
                               booster='gbtree',
                               n_estimators=1000,
                               objective='reg:squarederror',
                               max_depth=3,
                               early_stopping_rounds=5,
                               learning_rate=0.05,
                               colsample_bytree=0.7,
                               min_child_weight=4,
                               subsample=0.7)

    xgb_reg.fit(x_all, y_all,
        eval_set=[(x_all, y_all)],
        verbose=20)

    # create future data
    station_future = create_future_dates(station)
    station_future = create_features(station_future)
    station_future = station_future[station_future['isFuture']==True]
    station_future = station_future[features]

    # predict future temperature
    prediction = list(xgb_reg.predict(station_future[features]))
    return station_name, prediction


#%%
def predict_temperature_all_stations(dataframe):
    stations = dataframe['name'].unique()
    temperature_prediction = {}

    for station in stations:
        station_name, prediction = predict_temperature_by_station(station, temperature)
        temperature_prediction[station_name] = prediction

    return temperature_prediction


#%%
def create_temperature_forecast(temperature_dictonary):
    temperature_prediction = pd.DataFrame(temperature_dictonary)
    future = create_future_dates(temperature)
    future_index = future[future['isFuture']==True].index
    temperature_prediction['datetime'] = future_index

    return temperature_prediction


#%%
def return_temperature_file(temperature_prediction):
    predictions = temperature_prediction.mean(axis=1)
    predictions = predictions.rename("temperature")
    predictions = pd.DataFrame(predictions)
    #predictions.to_csv('data/temperature_forecast.csv')
    predictions.to_parquet('data/temperature_forecast.parquet.gzip', compression='gzip')

#%%
temperature_dictonary = predict_temperature_all_stations(temperature)
temperature_prediction = create_temperature_forecast(temperature_dictonary)
return_temperature_file(temperature_prediction)


#%% playground
mean_temp = temperature.groupby('recorded_at')['temperature'].mean()

#%%
mean_temp.index = pd.to_datetime(mean_temp.index)
mean_temp.plot()

#%%
predictions = temperature_prediction.mean(axis=1)
predictions.index = pd.date_range(start=mean_temp.index.max(), periods=len(predictions.index), freq='H')
predictions.plot()

#%%
mean_temp.plot()
predictions.plot()

#%%
