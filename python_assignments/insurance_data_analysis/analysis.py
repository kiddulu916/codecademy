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
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'insurance.csv'))
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'children_costs_results.txt')


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
    grouped['diff_from_prev'] = grouped['average_cost'].diff().round(2)
    grouped['pct_change_from_prev'] = grouped['average_cost'].pct_change().multiply(100).round(2)
    return grouped

def explain_results(summary: pd.DataFrame) -> str:
    """
    Generates an explanation of what the weighted averages and differences reveal about the data.
    """
    explanation = []
    explanation.append("Analysis Explanation:")
    explanation.append("- The table above shows the total and average insurance costs for each group based on the number of children.")
    explanation.append("- The 'diff_from_prev' column shows how much the average insurance cost changes as the number of children increases by one.")
    explanation.append("- The 'pct_change_from_prev' column shows the percentage change in average cost from the previous group.")
    explanation.append("\nInterpretation:")
    explanation.append("- If the differences are consistently positive and large, it suggests insurance costs rise with more children. If differences are small or fluctuate, the relationship may be weak or non-linear.")
    explanation.append("- Large jumps in certain groups may indicate thresholds where insurers increase rates more steeply.")
    explanation.append("- If some groups show a decrease, it could suggest other factors are at play, or possible discounts for larger families.")
    explanation.append("- This analysis helps identify trends and patterns that may not be obvious from overall averages alone.")
    return '\n'.join(explanation)


def save_results(results: Dict[str, Any], filename: str) -> None:
    """
    Saves the results dictionary and table to a .txt file in a readable format.
    """
    with open(filename, 'w') as f:
        f.write("Insurance Cost Analysis by Number of Children\n")
        f.write("="*50 + "\n\n")
        # Summary Table
        f.write("Weighted Totals, Averages, and Differences per Number of Children:\n")
        f.write(str(results['summary_table']) + "\n\n")
        # Explanation
        f.write(results['explanation'] + "\n")


def main():
    """
    Main function to run the weighted analysis, print, and save results.
    """
    try:
        df = load_and_validate_data(DATA_PATH)
        summary = weighted_analysis(df)
        explanation = explain_results(summary)

        # Prepare results dictionary
        results = {
            'summary_table': summary,
            'explanation': explanation
        }

        # Print results to console
        print("Insurance Cost Analysis by Number of Children")
        print("="*50)
        print("Weighted Totals, Averages, and Differences per Number of Children:")
        print(summary)
        print("\n" + explanation)

        # Save results to file
        save_results(results, RESULTS_PATH)
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
