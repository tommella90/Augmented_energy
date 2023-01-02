import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

#%%
df = pd.read_parquet('data/RES2_HC.parquet.gzip')
df.index = pd.to_datetime(df.index)
df = df.sort_index()

#%% PLOT AND CHECK OUTLIERS
fig, ax = plt.subplots(figsize=(15, 5))
df[['coefficient_value']].plot(ax=ax, label="coefficient")
plt.show()


#%%
def create_features(dataframe):
    dataframe['hours'] = dataframe.index.hour
    dataframe['days'] = dataframe.index.dayofweek
    dataframe['quarters'] = dataframe.index.quarter
    dataframe['months'] = dataframe.index.month
    dataframe['years'] = dataframe.index.year
    dataframe['dayofyear'] = dataframe.index.dayofyear
    return dataframe

df = create_features(df)

#%%
def add_lags(dataframe):
    target_map = dataframe['coefficient_value'].to_dict()
    dataframe['lag_1year'] = (dataframe.index - pd.Timedelta('364 days')).map(target_map)  ## same day last year
    dataframe['lag_2year'] = (dataframe.index - pd.Timedelta('728 days')).map(target_map)  ## same day last year
    dataframe['lag_3year'] = (dataframe.index - pd.Timedelta('1092 days')).map(target_map)  ## same day last year
    return dataframe

df = add_lags(df)


#%% VISUALIZE CONSUMPTION BY HOUR
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(data=df, x='hours', y='coefficient_value')
ax.set_title("MW by hour")


#%% VISUALIZE CONSUMPTION BY MONTH
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(data=df, x='months', y='coefficient_value', palette="Blues")
ax.set_title("MW by month")


#%% CREATE TRAIN AND TEST USING CROSS VALIDATION
n_split = 5
tss = TimeSeriesSplit(n_splits=n_split, test_size=24*365*1, gap=24)
df = df.sort_index()


#%% parameter tuning
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor

split_date = '2022-10-31'
df_training = df.drop(['profile_class', 'lag_2year'], axis = 1)
df_training = df_training.dropna()
train = df_training.loc[df_training.index < split_date]
test = df_training.loc[df_training.index >= split_date]


xgb = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:squarederror'],
              'learning_rate': [.03, .05, .07], #so called `eta` value
              'max_depth': [3, 5, 7],
              'min_child_weight': [4],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500, 750, 1000]}

xgb_grid = GridSearchCV(xgb,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(train, train)


#%%
print(xgb_grid.best_score_)
print(xgb_grid.best_params_)


#%% PLOT FOLDS
fig, axs = plt.subplots(n_split, 1, figsize=(15, 15), sharex=True)

fold = 0
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]
    train['coefficient_value'].plot(ax=axs[fold],
                                    label='Training Set',
                                    title=f'Data Train/Test Split Fold {fold}')
    test['coefficient_value'].plot(ax=axs[fold], label='Test Set')
    axs[fold].axvline(test.index.min(), color='black', ls='--')
    fold += 1
plt.show()



#%% FUNCTION TO TRAIN MODEL BY CLASS
import xgboost as xgb

def train_model_by_class(class_name):
    if class_name == 'BASE':
        df = pd.read_parquet('models/RES11_BASE.parquet.gzip')
    elif class_name == "HC":
        df = pd.read_parquet('models/RES2_HC.parquet.gzip')
    elif class_name == "HP":
        df = pd.read_parquet('models/RES2_HP.parquet.gzip')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # time series kfold
    n_split = 5
    tss = TimeSeriesSplit(n_splits=n_split, test_size=24*365*1, gap=24)

    fold = 0
    preds, scores = [], []
    for train_idx, validation_idx in tss.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[validation_idx]

        # create features
        train = create_features(train)
        train = add_lags(train)
        test = create_features(test)
        test = add_lags(test)

        # split test and train
        features = ['temperature', 'hours', 'days', 'quarters', 'months', 'years', 'dayofyear',
                    'lag_1year', 'lag_2year']
        target = 'coefficient_value'

        x_train = train[features]
        y_train = train[target]
        x_test = test[features]
        y_test = test[target]

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

        xgb_reg.fit(x_train, y_train,
                eval_set=[(x_train, y_train), (x_test, y_test)],
                verbose=20)

        y_pred = xgb_reg.predict(x_test)
        preds.append(y_pred)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(score)
        #print(f'Fold {fold} RMSE: {score}')
        fold += 1

        print(f"Average score across folds {np.mean(scores): 0.4f}")
        print(f"Fold scores: {scores}")

        with open(f"data/model_xgb_{class_name}.pickle", "wb") as f:
            pickle.dump(xgb_reg, f)

        return xgb_reg


#%% TRAIN MODEL BY CLASS
for class_group in ["BASE", "HC", "HP"]:
    xgb_reg = train_model_by_class(class_group)


#%% FEATURE IMPORTANCE


fi = pd.DataFrame(data=xgb_reg.feature_importances_,
                  index=xgb_reg.feature_names_in_,
                  columns=['importance'])

fi.sort_values('importance').plot(kind='barh', title='Feature Importance')


#%% PLOT ENTIRE DATASET AND PREDICTION (last test set)
df_plot = df.merge(test[['prediction']], how="left", left_index=True, right_index=True)

ax = df_plot[['coefficient_value']].plot(figsize=(15, 5))
df_plot[['prediction']].plot(ax=ax, style='.')
plt.legend(['Truth data', 'prediction'])
ax.set_title("Raw data and prediction")


#%% EVALUATE
rmse = np.sqrt(mean_squared_error(test['coefficient_value'], test['prediction']))
print(f"RMSE score on test set: {rmse:0.2f}")


#%% CHECK WORST DAYS
test['error'] = np.abs(test['coefficient_value'] - test['prediction'])
test['date'] = test.index.date
test.groupby('date')['error'].mean().sort_values(ascending=False)


#%% PREDICT INTO THE FUTURE
# RETRAIN THE MODEL WITH THE WHOLE DATASET
# choose the n_estimators the moment it starts to overfit
features = ['temperature', 'hours', 'days', 'quarters', 'months', 'years', 'dayofyear',
            'lag_1year', 'lag_2year']
target = 'coefficient_value'

x_all = df[features]
y_all = df[target]

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


#%% create a new dataframe in the future
future_boundary = df.index.max() + pd.Timedelta('1 y')
future = pd.date_range(df.index.max(), future_boundary, freq='1h')


#%%
future_df = pd.DataFrame(index=future)
future_df['isFuture'] = True
df['isFuture'] = False
df_and_future = pd.concat([df, future_df])

# create features and lags
df_and_future = create_features(df_and_future)
df_and_future = add_lags(df_and_future)

future_with_features = df_and_future.query('isFuture').copy()


#%% PREDICT THE FUTURE
future_with_features['prediction'] = xgb_reg.predict(future_with_features[features])
future_with_features['prediction'].plot(figsize=(10, 5),
                                        color="blue",
                                        ms=1, lw=1, title="Future prediction")




