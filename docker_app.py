#%%
from AugmentedEnergyPredictor import AugmentedEnergyPrediction


def main():
    """
    Choose a consumption class: \n-BASE\n-HP\n-HC"
    """
    try:
        class_chosen = input("Choose a consumption class: \n-BASE\n-HP\n-HC\n")
        energy_usage_calculation = AugmentedEnergyPrediction(class_chosen)

        day_of_prediction = input("Choose a day beteen 2022-10-31 and 2023-10-30 (YYYY-MM-DD)\n")
        prediction = round(energy_usage_calculation.predict_day_energy_consumption(day_of_prediction), 4)
        dict_hours = prediction.to_dict()
        print(f"Energy consumption prediction for {day_of_prediction} is:\n {dict_hours}")
    except:
        print("The day is not valid. Please choose a day between 2022-10-31 and 2023-10-30 (YYYY-MM-DD)")

main()

#%%