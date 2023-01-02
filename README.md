# task, TOMMASO RAMELLA

Hello Eric, here are my solutions to the task, and a brief explenation.

First way, you can simply run a little demo I put in the docker file (here). It returns a daily prediction (up to 1 year), giving the predicted consumption per each hour. 
(I'm not very expert with Docker, so in case it doesnt' work you can check the Git repo)

In the Github repository you can file the folliwing files that I used to prepare the data and train a model: 
1) clean_data.py
A few basic data cleaning and merging to combine the datasets

2) temperature_forecasting.py
I use an xgboost alghoritm to forecast temperature up to 1 year after the last day available. I preict temperature for each station separately, and then take the average oer each hour. I'll use this forecasted temperature as 
a feature to train a general model later. 

3) eda_modelTraining.py
Here I do exploratory data analysis and I train a model to predict energy consumption up to 1 year. 
It returns 3 model, one per each consumption category, stored in /models. 

4) AugmentedEnergyPredictor.py
This is the py package that creates 4 main ready functions (you can check directly the documentation with 
help(AugmentedEnergyPrediction)
):
	- predict_year_energy_consumption: returns a dataframe containing all the energy-consumption predictions for one year (hoursly based). 
	- predict_day_energy_consumption: returns a dataframe containing a specific day consuption prediction (hourly based). The argument is a specifica date (YYYY-MM-DD)
	- plot_year_prediction: plot the past and predicted energy consumption
	- plot_day_prediction: plot the daily prediction. The argument is a specific date (YYYY-MM-DD)

You can use this module to explore the outputs. To start, you need to choose one of the three energy classes ("BASE", "HP", "HC")
eg:

energy_prediction = AugmentedEnergyPrediction(consumption_group="BASE")

and then calculate a daily prediction in the future: 
energy_prediction.plot_day_prediction("2023-01-01")


5) input_function
You can also run this py file directly from the command line and follow the instructions

