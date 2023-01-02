import pandas as pd

#%%
actual = pd.read_csv("data/weather_actuals.csv.gz", compression='gzip')
#actual['recorded_at'] = pd.to_datetime(actual['recorded_at'])

profiles = pd.read_csv("data/load_profiles.csv.gz", compression='gzip')
#profiles['delivery_from'] = pd.to_datetime(profiles['delivery_from'])

#%%
stations = pd.read_csv("data/weather_stations.csv")
actual = actual[actual['station_id'].isin(list(stations['id'].unique()))]
temperature_by_station = pd.merge(actual, stations, left_on='station_id', right_on='id', how='left')
temperature_by_station = temperature_by_station.sort_values(by="recorded_at")
temperature_by_station.tail(10)


#%% SELECT THE CLASS
base_consumers = profiles.loc[profiles['profile_class']=='RES11_BASE']
high_consumers = profiles.loc[profiles['profile_class']=='RES2_HC']
low_consumers = profiles.loc[profiles['profile_class']=='RES2_HP']
consumers = [base_consumers, high_consumers, low_consumers]

#%% CLEAN DATASET
temperature_by_station = temperature_by_station[['recorded_at', 'station_id', 'temperature']]
temperature_by_station = temperature_by_station.sort_values(by='recorded_at')


#%%GET AVERAGE TEMPERATURE AND MERGE WITH THE RECORDS
temperature = temperature_by_station.groupby('recorded_at')['temperature'].mean()

consumers_temperature = []
for dataframe in consumers:
    dataframe = pd.merge(temperature, dataframe,
               left_on='recorded_at',
               right_on='delivery_from',
               how='right',
               indicator=False)
    dataframe.set_index('delivery_from', inplace=True)
    dataframe.to_parquet(f"{dataframe['profile_class'].unique()[0]}.parquet.gzip", compression='gzip')
    consumers_temperature.append(dataframe)
    #consumers_temperature.append(dataframe)



#%%
