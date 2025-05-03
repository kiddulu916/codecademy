"""
insurance_data_analysis/analysis.py

Performs a weighted analysis of insurance costs in relation to the number of children using the insurance.csv dataset.
- Computes weighted averages and totals per number of children (weights = group size)
- Analyzes differences and percentage changes between groups
- Performs regression analysis
- Explains what the calculations reveal about the data
- Prints and saves results to children_costs_results.txt

Author: [Your Name]
Date: 2025-04-27
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LinearRegression

# Path to the insurance dataset (relative to this script)
DATA_PATH = 'insurance.csv'
RESULTS_PATH = 'children_costs_results.txt'

def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Loads the insurance dataset and validates required columns.
    Handles missing or malformed data.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

    required_cols = {'children', 'charges'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    df = df.dropna(subset=['children', 'charges'])
    df = df[(pd.to_numeric(df['children'], errors='coerce').notnull()) & (pd.to_numeric(df['charges'], errors='coerce').notnull())]
    df['children'] = df['children'].astype(int)
    df['charges'] = df['charges'].astype(float)
    return df

# Basic statistics
def age_analysis(df: pd.DataFrame) -> pd.DataFrame:
    total_num_of_patients = len(df)
    youngest_patient = df['age'].min()
    oldest_patient = df['age'].max()
    
    # Smokers and non-smokers
    total_num_of_smokers = (df['smoker'] == 'yes').sum()
    total_num_of_nonsmokers = (df['smoker'] == 'no').sum()

    diff_in_total_smokers_nonsmokers = total_num_of_smokers - total_num_of_nonsmokers

    # With children vs without children
    total_num_with_children = (df['children'] > 0).sum()
    total_num_without_children = (df['children'] == 0).sum()
    diff_in_total_with_without_children = total_num_with_children - total_num_without_children

    # Male vs Female patients
    num_of_male_patients = (df['sex'] == 'male').sum()
    num_of_female_patients = (df['sex'] == 'female').sum()
    diff_in_males_to_females = num_of_male_patients - num_of_female_patients

    # Male and Female Smokers
    num_of_male_smokers = df[(df['sex'] == 'male') & (df['smoker'] == 'yes')].shape[0]
    num_of_female_smokers = df[(df['sex'] == 'female') & (df['smoker'] == 'yes')].shape[0]
    diff_in_male_to_female_smokers = num_of_male_smokers - num_of_female_smokers

    # Male and Female with children
    num_of_males_with_children = df[(df['sex'] == 'male') & (df['children'] > 0)].shape[0]
    num_of_females_with_children = df[(df['sex'] == 'female') & (df['children'] > 0)].shape[0]
    diff_in_male_to_female_with_children = num_of_males_with_children - num_of_females_with_children

    age_average = df['age'].mean()
    smoker_age_average = df[df['smoker'] == 'yes']['age'].mean()
    nonsmoker_age_average = df[df['smoker'] == 'no']['age'].mean()
    with_children_age_average = df[df['children'] > 0]['age'].mean()
    no_children_age_average = df[df['children'] == 0]['age'].mean()

    # Print results
    age_analysis_results = f"""Total number of patients: {total_num_of_patients}
Patients ages range from: {youngest_patient} - {oldest_patient}

There are {num_of_male_patients} male patients.
{num_of_male_smokers} of those males are smokers and {num_of_males_with_children} of them have at least 1 child.

There are {num_of_female_patients} female patients.
{num_of_female_smokers} of those females are smokers and {num_of_females_with_children} of them have at least 1 child.

Average age of patients overall: {str(int(age_average))}

Average age of patients that smoke: {str(int(smoker_age_average))}
Average age of patients that don't smoke: {str(int(nonsmoker_age_average))}

Average age of patients with at least 1 child: {str(int(with_children_age_average))}
Average age of patients with 0 children: {str(int(no_children_age_average))}\n"""
    
    return age_analysis_results

def find_totals_and_averages(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate total and average insurance costs overall
    total_insurance_cost = df['charges'].sum()
    average_total_cost = df['charges'].mean()

    # Rounding
    rounded_total_cost = round(total_insurance_cost, 2)
    rounded_average_cost = round(average_total_cost, 2)

    # Print results
    totals_and_averages_results = f"""The total amount spent on insurance is: ${rounded_total_cost}
The average insurance cost of overall costs is: ${rounded_average_cost}\n"""
    
    return totals_and_averages_results

def analyze_costs_by_sex(df: pd.DataFrame) -> pd.DataFrame:
    # Group by sex and sum/mean the charges
    costs_by_sex = df.groupby('sex')['charges'].agg(['sum', 'mean'])

    # Extract values
    total_male_cost = costs_by_sex.loc['male', 'sum']
    total_female_cost = costs_by_sex.loc['female', 'sum']

    average_male_cost = costs_by_sex.loc['male', 'mean']
    average_female_cost = costs_by_sex.loc['female', 'mean']

    # Differences
    difference_in_total_costs = total_male_cost - total_female_cost
    difference_in_average_costs = average_male_cost - average_female_cost

    # Rounding
    rounded_total_male_cost = round(total_male_cost, 2)
    rounded_total_female_cost = round(total_female_cost, 2)
    rounded_average_male_cost = round(average_male_cost, 2)
    rounded_average_female_cost = round(average_female_cost, 2)
    rounded_total_difference = round(difference_in_total_costs, 2)
    rounded_average_difference = round(difference_in_average_costs, 2)

    # Print results
    costs_by_sex_results = f"""Total insurance costs for males is: ${rounded_total_male_cost}
Total insurance costs for females is: ${rounded_total_female_cost}
Difference of total costs between males and females is: ${rounded_total_difference}

Average insurance costs for males is: ${rounded_average_male_cost}
Average insurance costs for females is: ${rounded_average_female_cost}
Difference of average costs between males and females is: ${rounded_average_difference}\n"""
    
    return costs_by_sex_results

def analyze_smoker_costs(df: pd.DataFrame) -> pd.DataFrame:
    # Group by smoker and sex, then calculate total and average charges
    smoker_costs_summary = df.groupby(['smoker', 'sex'])['charges'].agg(total='sum', average='mean').round(2)

    # Smokers vs Non-Smokers overall
    overall_smoker_costs = df.groupby('smoker')['charges'].agg(total='sum', average='mean').round(2)
    rounded_total_smoker_cost = overall_smoker_costs.loc['yes', 'total']
    rounded_total_nonsmoker_cost = overall_smoker_costs.loc['no', 'total']
    rounded_average_smoker_cost = overall_smoker_costs.loc['yes', 'average']
    rounded_average_nonsmoker_cost = overall_smoker_costs.loc['no', 'average']

    # Differences
    rounded_smoker_total_difference = round(rounded_total_nonsmoker_cost - rounded_total_smoker_cost, 2)
    rounded_smoker_average_difference = round(rounded_average_smoker_cost - rounded_average_nonsmoker_cost, 2)

    # Male Smokers vs Male Non-Smokers
    total_male_smoker_cost = smoker_costs_summary.loc[('yes', 'male'), 'total']
    total_male_nonsmoker_cost = smoker_costs_summary.loc[('no', 'male'), 'total']
    average_male_smoker_cost = smoker_costs_summary.loc[('yes', 'male'), 'average']
    average_male_nonsmoker_cost = smoker_costs_summary.loc[('no', 'male'), 'average']

    rounded_male_smoker_total_difference = round(total_male_smoker_cost - total_male_nonsmoker_cost, 2)
    rounded_male_smoker_average_difference = round(average_male_smoker_cost - average_male_nonsmoker_cost, 2)

    # Female Smokers vs Female Non-Smokers
    total_female_smoker_cost = smoker_costs_summary.loc[('yes', 'female'), 'total']
    total_female_nonsmoker_cost = smoker_costs_summary.loc[('no', 'female'), 'total']
    average_female_smoker_cost = smoker_costs_summary.loc[('yes', 'female'), 'average']
    average_female_nonsmoker_cost = smoker_costs_summary.loc[('no', 'female'), 'average']

    rounded_female_smoker_total_difference = round(total_female_nonsmoker_cost - total_female_smoker_cost, 2)
    rounded_female_smoker_average_difference = round(average_female_smoker_cost - average_female_nonsmoker_cost, 2)

    # Male vs Female Smokers
    diff_male_female_smoker_total = round(total_male_smoker_cost - total_female_smoker_cost, 2)
    diff_male_female_smoker_average = round(average_male_smoker_cost - average_female_smoker_cost, 2)

    # Male vs Female Non-Smokers
    diff_male_female_nonsmoker_total = round(total_female_nonsmoker_cost - total_male_nonsmoker_cost, 2)
    diff_male_female_nonsmoker_average = round(average_female_nonsmoker_cost - average_male_nonsmoker_cost, 2)

    # Print results
    smokers_results = f"""Total smoker costs: ${rounded_total_smoker_cost}
Total non-smoker costs: ${rounded_total_nonsmoker_cost}
Difference in total costs between smokers and non-smokers: ${rounded_smoker_total_difference}

Average smoker cost: ${rounded_average_smoker_cost}
Average non-smoker cost: ${rounded_average_nonsmoker_cost}
Difference in average costs between smokers and non-smokers: ${rounded_smoker_average_difference}

Male smoker vs male non-smoker total cost difference: ${rounded_male_smoker_total_difference}0
Male smoker vs male non-smoker average cost difference: ${rounded_male_smoker_average_difference}

Female smoker vs female non-smoker total cost difference: ${rounded_female_smoker_total_difference}
Female smoker vs female non-smoker average cost difference: ${rounded_female_smoker_average_difference}0

Male vs Female smoker total cost difference: ${diff_male_female_smoker_total}
Male vs Female smoker average cost difference: ${diff_male_female_smoker_average}

Male vs Female non-smoker total cost difference: ${diff_male_female_nonsmoker_total}
Male vs Female non-smoker average cost difference: ${diff_male_female_nonsmoker_average}0\n"""
    
    return smokers_results

def weighted_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes weighted totals and averages per number of children.
    Also computes differences and percentage changes between groups.
    Weights are the group sizes (counts).
    Returns a summary DataFrame.
    """
    grouped = round(df.groupby('children')['charges'].agg(['count', 'sum', 'mean']).reset_index(), 2)
    grouped.rename(columns={'count': 'num_records', 'sum': 'total_cost', 'mean': 'average_cost'}, inplace=True)
    
    # Calculate differences and percent changes
    grouped['diff_prev'] = grouped['average_cost'].diff().round(2)
    grouped['pct_of_change'] = grouped['average_cost'].pct_change().multiply(100).round(2)

    totals = []
    averages = []
    num_of_children = []
    
    for num in grouped['children']:
        str(num_of_children.append(num))

    for num in grouped['total_cost']:
        str(totals.append(num))

    for num in grouped['average_cost']:
        str(averages.append(num))
    
    children_results = f"""Total insurance cost for people with {num_of_children[0]} children: ${totals[0]}
Average insurance cost for people with {num_of_children[0]} children: ${averages[0]}
    
Total insurance cost for people with {num_of_children[1]} child: ${totals[1]}
Average insurance cost for people with {num_of_children[1]} child: ${averages[1]}
    
Total insurance cost for people with {num_of_children[2]} children: ${totals[2]}
Average insurance cost for people with {num_of_children[2]} children: ${averages[2]}
    
Total insurance cost for people with {num_of_children[3]} children: ${totals[3]}
Average insurance cost for people with {num_of_children[3]} children: ${averages[3]}
    
Total insurance cost for people with {num_of_children[4]} children: ${totals[4]}
Average insurance cost for people with {num_of_children[4]} children: ${averages[4]}
    
Total insurance cost for people with {num_of_children[5]} children: ${totals[5]}
Average insurance cost for people with {num_of_children[5]} children: ${averages[5]}\n"""
    
    return children_results

# Main function
def main():
    """
    Main function to run the weighted analysis, print, and save results.
    """
    try:
        dataset = load_and_validate_data(DATA_PATH)
        ages = age_analysis(dataset)
        overall = find_totals_and_averages(dataset)
        sex = analyze_costs_by_sex(dataset)
        smokers = analyze_smoker_costs(dataset)
        children = weighted_analysis(dataset)

        # Print results to console
        print("- Age Analysis\n")
        print(ages + "\n")
        print("""**With this analysis of ages that some of the data in some of the 
calculations is imbalanced there for not completely accurate.\n""") 
        print("- Overall Insurance Costs Totals and Averages\n")
        print(overall + "\n")
        print("- Insurance Cost Analysis by Sex\n")
        print(sex + "\n")
        print("- Insurance Cost Analysis by Smokers and Non-Smokers\n")
        print(smokers + "\n")
        print("- Insurance Cost Analysis by Number of Children\n")
        print(children)

        def save_results(dataset, ages, overall, sex, smokers, children, save_path):
            """
            Saves the results dictionary and table to a .txt file in a readable format.
            """
            with open(save_path, 'w') as f:
                # Age analysis
                f.write("- Age Analysis\n\n")
                f.write(ages)
                f.write("\n**With this analysis of ages that some of the data in some of the calculations is imbalanced there for not completely accurate.\n\n")
                # Analysis of overall totals and averages
                f.write("- Overall Insurance Costs Totals and Averages\n\n")
                f.write(overall + "\n")
                # Analysis of insurance costs by sex
                f.write("- Insurance Cost Analysis by Sex\n\n")
                f.write(sex + "\n")
                # Analysis of insurance costs by smokers
                f.write("- Insurance Cost Analysis by Smokers and Non-Smokers\n\n")
                f.write(smokers + "\n")
                # Analysis of insurance costs by 
                f.write("- Insurance Cost Analysis by Number of Children\n\n")
                f.write(children + "\n\n")
        
        # Save results to file
        save_results(dataset, ages, overall, sex, smokers, children, RESULTS_PATH)
        print(f"\nResults saved to {RESULTS_PATH}")

    except Exception as e:
        print(f"Error: {e}")

def test_analysis_functions():
    """
    Simple test to verify weighted analysis functions on a small sample DataFrame.
    """
    test_data = pd.DataFrame({
        'children': [0, 1, 2, 2, 3],
        'charges': [1000, 2000, 3000, 4000, 5000]
    })
    summary = weighted_analysis(test_data)
    assert not summary.empty, "Summary table should not be empty."
    print("Test passed: Weighted analysis functions work as expected.")


if __name__ == "__main__":
    main()
    # Uncomment to run test
    # test_analysis_functions()
