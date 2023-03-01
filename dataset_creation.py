# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import json

# %%


# %%
def read_mls_salaries(data_path, year):

    salaries_year = pd.read_csv(os.path.join(data_path, f"mls-salaries-{year}.csv"))

    salaries_year["salary"] = salaries_year["guaranteed_compensation"]

    # salaries_year = salaries_year[salaries_year["position"].str.contains("F") 
                                #   & (~salaries_year["position"].isna())]

    salaries_year = salaries_year[["first_name", "last_name", "salary"]]

    return salaries_year

def read_stats_data(data_path, year):

    stats_year = pd.read_json(os.path.join(data_path, f"data_{year}.json"))

    stats_year["age_group"] = pd.cut(stats_year["age"], 
                         bins=[20, 27, 30, 35, 100], 
                         labels=["20-27", "27-30", "30-35", ">35"])

    
    stats_year["goals_per_game"] = (stats_year["goal"] + stats_year["assistTotal"]) / stats_year["apps"]
    stats_year["shots_per_game"] = stats_year["shotsPerGame"] # / stats_year["apps"]
    stats_year["minutes_per_game"] = stats_year["minsPlayed"] / stats_year["apps"]

    stats_year = stats_year.rename(columns={"teamName": "team_name",
                                            "passSuccess": "pass_success",
                                            "firstName": "first_name",
                                            "lastName": "last_name"})

    stats_year = stats_year[["age_group", 
                             "team_name", 
                             "shots_per_game", 
                             "goals_per_game",
                             "pass_success",
                             "rating",
                             "minutes_per_game",
                             "first_name",
                             "last_name"]]


    # stats_year = pd.get_dummies(stats_year, columns=["age_group", "team_name"])


    return stats_year



# %%
def merge_datasets(year, data_path):
    merge_on = ["last_name", "first_name"]


    salaries_path = os.path.join(data_path, "csvs")

    salaries_year = read_mls_salaries(salaries_path, year)
    salaries_next_year = read_mls_salaries(salaries_path, year + 1)



    # Merge the two datasets
    salaries_merged = pd.merge(salaries_year, 
                               salaries_next_year, 
                               on=merge_on, 
                               how="inner",
                               suffixes=("", "_next_year"))


    added_salary = (salaries_merged["salary_next_year"] - salaries_merged["salary"]) / salaries_merged["salary"] 
    salaries_merged["added_salary"] = added_salary

    salaries_merged = salaries_merged[salaries_merged["added_salary"] > -0.1]
    salaries_merged["T"] = salaries_merged["added_salary"] > 0.1
    

    salaries_merged["salary_next_year"] = np.log(salaries_merged["salary_next_year"])
    salaries_merged["salary"] = np.log(salaries_merged["salary"])


    stats_path = os.path.join(data_path, "jsons")

    stats_year = read_stats_data(stats_path, year)
    stats_next_year = read_stats_data(stats_path, year + 1)


    # Merge the two datasets
    stats_merged = pd.merge(stats_year,
                            stats_next_year,    
                            on=merge_on,
                            how="inner",
                            suffixes=("", "_next_year"))

    
    a1 = salaries_merged.columns
    a2 = stats_merged.columns
    a3 = np.intersect1d(a1, a2)
    a3 = a3[~np.isin(a3, merge_on)]

    
    assert len(a3) == 0, "There are columns with the same name in both datasets"

    # Merge the two datasets
    merged = pd.merge(salaries_merged,
                      stats_merged,
                        on=merge_on,
                        # left_on=["last_name", "first_name"],
                        # right_on=["lastName", "firstName"],
                        how="inner")

    return merged
    

def create_dataset():
    df = pd.DataFrame()

    for year in range(2013, 2018):

        # df = merge_datasets(year, "archive")
        df = df.append(merge_datasets(year, "archive"))

    return df

# %%
if __name__ == "__main__":
    df = create_dataset()

    print(df.head())
