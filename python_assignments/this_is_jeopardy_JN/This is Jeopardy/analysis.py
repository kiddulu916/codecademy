"""
This script analyzes Jeopardy data to find patterns and insights in questions and answers.
It performs several key analyses:
1. Data cleaning and preparation
2. Question filtering based on keywords
3. Value analysis of questions
4. Answer frequency analysis
"""

import pandas as pd
pd.set_option('display.max_colwidth', 50)  # Set maximum column width for better readability

# Load and prepare the Jeopardy dataset
def load_and_prepare_data():
    """
    Load the Jeopardy dataset and prepare it for analysis by:
    1. Reading the CSV file
    2. Cleaning column names by removing whitespace
    3. Converting value column to numeric format
    """
    # Import the dataset
    jeopardy_data = pd.read_csv("jeopardy.csv")
    
    # Clean column names by removing leading whitespace
    print("\nOriginal column names with whitespace:")
    print(jeopardy_data[' Answer'].head(5))
    
    # Rename columns to remove whitespace
    jeopardy_data = jeopardy_data.rename(columns = {
        " Air Date": "Air Date", 
        " Round" : "Round", 
        " Category": "Category", 
        " Value": "Value", 
        " Question": "Question", 
        " Answer": "Answer"
    })
    print("\nColumn names after whitespace removal:")
    print(jeopardy_data['Answer'].head(5))
    
    # Convert value column to numeric format for analysis
    print("\nConverting value column to numeric format...")
    jeopardy_data['Float Value'] = jeopardy_data['Value'].str.replace('$', '', regex=False)\
        .str.replace(',', '', regex=False)\
        .str.replace('no value', '0', regex=False)\
        .astype(float)
    print("\nFirst 5 rows of converted float values:")
    print(jeopardy_data['Float Value'].head(5))
    
    return jeopardy_data

def filter_questions_by_words(data, words):
    """
    Filter questions based on specific keywords.
    
    Args:
        data: DataFrame containing Jeopardy data
        words: List of keywords to search for in questions
        
    Returns:
        Filtered DataFrame containing only questions that match all keywords
    """
    # Convert search words to lowercase for case-insensitive matching
    lowercase_words = [word.lower() for word in words]
    
    # Filter questions that contain all specified keywords
    for word in lowercase_words:
        pattern = r'\b' + word + r'\b'  # Use word boundaries for exact matching
        data = data[data['Question'].str.lower().str.contains(pattern)]
    
    return data

def calculate_average_value(data, words):
    """
    Calculate the average value of questions containing specific keywords.
    
    Args:
        data: DataFrame containing Jeopardy data
        words: List of keywords to filter questions
        
    Returns:
        Average value of matching questions, rounded to 2 decimal places
    """
    # Filter questions containing the specified keywords
    filtered = filter_questions_by_words(data, words)
    
    # Calculate and return the average value
    return round(filtered['Float Value'].mean(), 2)

def analyze_answer_frequencies(data, words):
    """
    Analyze the frequency of answers for questions containing specific keywords.
    
    Args:
        data: DataFrame containing Jeopardy data
        words: List of keywords to filter questions
        
    Returns:
        String containing the most common answer and its frequency
    """
    # Filter questions containing the specified keywords
    filtered = filter_questions_by_words(data, words)
    
    # Count occurrences of each answer
    answer_counts = {}
    for answer in filtered['Answer']:
        if answer in answer_counts:
            answer_counts[answer] += 1
        else:
            answer_counts[answer] = 1
    
    # Sort answers by frequency in descending order
    sorted_answer_counts = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Display top 5 most common answers
    print(f'\nTop 5 most common answers for questions containing {", ".join(words)}:')
    for answer, count in sorted_answer_counts[:5]:
        print(f"The answer: '{answer}' appeared {count} times")
    
    # Return information about the most common answer
    most_common_answer = sorted_answer_counts[0]
    return f"\nThe most common answer is: '{most_common_answer[0]}' with {most_common_answer[1]} occurrences"

# Main execution
if __name__ == "__main__":
    # Load and prepare the data
    print("\nLoading and preparing Jeopardy data...")
    jeopardy_data = load_and_prepare_data()
    
    # Test the question filtering functionality
    print("\nTesting question filtering with 'King' and 'England':")
    filtered = filter_questions_by_words(jeopardy_data, ["King", "England"])
    print("\nFiltered questions containing 'King' and 'England':")
    print(filtered)
    
    # Test the average value calculation
    print("\nCalculating average value of questions containing 'King' and 'England':")
    avg_value = calculate_average_value(jeopardy_data, ["King", "England"])
    print(f"Average value: ${avg_value}")
    
    # Test the answer frequency analysis
    print("\nAnalyzing answers for questions containing 'King' and 'England':")
    print(analyze_answer_frequencies(jeopardy_data, ["King", "England"]))