#%%
from AugmentedEnergyPredictor import AugmentedEnergyPrediction

def main():
    """
    Choose a consumption class: \n-BASE\n-HP\n-HC"
    """
    class_chosen = input("Choose a consumption class: \n-BASE\n-HP\n-HC\n")
    energy_usage_calculation = AugmentedEnergyPrediction(class_chosen)

    next_function = input("""
    What do you want to do: \n(A)Predict year energy consumption\n(B)Predict day energy consumption\n(C)Plot 1-year energy consumption prediction \n(D)Plot day energy consumption prediction\n\n\n\n""")

    if next_function == "A":
        prediction = energy_usage_calculation.predict_year_energy_consumption()
        print(prediction)
        return prediction

    elif next_function== "B":
        day = input("Choose a day beteen 2022-10-31 and 2023-10-30 (YYYY-MM-DD)\n")
        prediction = energy_usage_calculation.predict_day_energy_consumption(day)
        print(prediction)
        return prediction

    elif next_function == "C":
        return energy_usage_calculation.plot_year_prediction()

    elif next_function == "D":
        day = input("Choose a day beteen 2022-10-31 and 2023-10-30 (YYYY-MM-DD)\n")
        return energy_usage_calculation.plot_day_prediction(day)

main()


# %%
