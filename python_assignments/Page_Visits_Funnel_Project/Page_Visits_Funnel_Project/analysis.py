import pandas as pd

# Import dataframes
visits = pd.read_csv('visits.csv',
                     parse_dates=[1])
cart = pd.read_csv('cart.csv',
                   parse_dates=[1])                   
checkout = pd.read_csv('checkout.csv',
                       parse_dates=[1])
purchase = pd.read_csv('purchase.csv',
                       parse_dates=[1])

# Inspect dataframes
print("\nSample data from visits.csv:")
print(visits.head(5))
print("\nSample data from cart.csv:")
print(cart.head(5))
print("\nSample data from checkout.csv:")
print(checkout.head(5))
print("\nSample data from purchase.csv:")
print(purchase.head(5))

# Function to merge dataframes and reset index
def merge_dataframes(df1, df2):
    merged_df = pd.merge(df1, df2, how='left').reset_index()
    return merged_df

# Function find the length of a dataframe
def find_length(df):
    return df.size

# Function to count the number of null values in a column
def count_nulls(df, column):
    return df[column].isnull().sum()

# Function to calculate the percentage of null values in a column
def calculate_percentage_nulls(df, column):
    return (count_nulls(df, column) / find_length(df)) * 100

# Left merge visits and cart
visits_cart = merge_dataframes(visits, cart)
print("\nMerged visits and cart data (first 5 rows):")
print(visits_cart.head(5))

# Find the length of visits_cart
length_of_visits_cart = find_length(visits_cart)
print(f"\nTotal number of records in merged visits and cart: {length_of_visits_cart}")

# Find how many timestamps in the cart_time column are null
null_cart_time = count_nulls(visits_cart, 'cart_time')
print(f"Number of users who visited but didn't add to cart: {null_cart_time}")

# Calculate the percentages of null cart times
percentage_null_cart_time = calculate_percentage_nulls(visits_cart, 'cart_time')
print(f"Percentage of users who visited but didn't add to cart: {percentage_null_cart_time:.2f}%")

# Left merge cart and checkout
cart_checkout = merge_dataframes(cart, checkout)
print("\nMerged cart and checkout data (first 5 rows):")
print(cart_checkout.head(5))

# Find the length of cart_checkout
length_of_cart_checkout = find_length(cart_checkout)
print(f"\nTotal number of records in merged cart and checkout: {length_of_cart_checkout}")

# Find how many timestamps in the checkout_time column are null
null_cart_checkout_time = count_nulls(cart_checkout, 'checkout_time')
print(f"Number of users who added to cart but didn't proceed to checkout: {null_cart_checkout_time}")

# Calculate the percentages of null checkout times
percentage_null_cart_checkout_time = calculate_percentage_nulls(cart_checkout, 'checkout_time')
print(f"Percentage of users who added to cart but didn't proceed to checkout: {percentage_null_cart_checkout_time:.2f}%")

# Left merge checkout and purchase
checkout_purchase = merge_dataframes(checkout, purchase)
print("\nMerged checkout and purchase data (first 5 rows):")
print(checkout_purchase.head(5))

# Find the length of checkout_purchase
length_of_checkout_purchase = find_length(checkout_purchase)
print(f"\nTotal number of records in merged checkout and purchase: {length_of_checkout_purchase}")

# Find how many timestamps in the purchase_time column are null
null_checkout_purchase_time = count_nulls(checkout_purchase, 'purchase_time')
print(f"Number of users who proceeded to checkout but didn't complete purchase: {null_checkout_purchase_time}")

# Calculate the percentages of null purchase times
percentage_null_checkout_purchase_time = calculate_percentage_nulls(checkout_purchase, 'purchase_time')
print(f"Percentage of users who proceeded to checkout but didn't complete purchase: {percentage_null_checkout_purchase_time:.2f}%")

# Merge all 4 dataframes
all_data = merge_dataframes(visits_cart, checkout_purchase)
print("\nMerged data from all stages (first 5 rows):")
print(all_data.head(5))

# Find the length of all_data
length_of_all_data = find_length(all_data)
print(f"\nTotal number of records in combined dataset: {length_of_all_data}")

# Find how many timestamps in the cart_time column are null
null_cart_time = count_nulls(all_data, 'cart_time')
print(f"Number of users who visited but didn't add to cart (from combined data): {null_cart_time}")

# Find how many timestamps in the checkout_time column are null
null_checkout_time = count_nulls(all_data, 'checkout_time')
print(f"Number of users who added to cart but didn't proceed to checkout (from combined data): {null_checkout_time}")

# Find how many timestamps in the purchase_time column are null
null_purchase_time = count_nulls(all_data, 'purchase_time')
print(f"Number of users who proceeded to checkout but didn't complete purchase (from combined data): {null_purchase_time}")

# Calculate the percentages of null cart times
percentage_null_cart_time = calculate_percentage_nulls(all_data, 'cart_time')
print(f"Percentage of users who visited but didn't add to cart (from combined data): {percentage_null_cart_time:.2f}%")

# Calculate the percentages of null checkout times
percentage_null_checkout_time = calculate_percentage_nulls(all_data, 'checkout_time')
print(f"Percentage of users who added to cart but didn't proceed to checkout (from combined data): {percentage_null_checkout_time:.2f}%")

# Calculate the percentages of null purchase times
percentage_null_purchase_time =  calculate_percentage_nulls(all_data, 'purchase_time')
print(f"Percentage of users who proceeded to checkout but didn't complete purchase (from combined data): {percentage_null_purchase_time:.2f}%")

# Compare null percentages to find the highest percentage of users not completing a purchase
highest_null_percentage = max(percentage_null_cart_time, percentage_null_checkout_time, percentage_null_purchase_time)
print(f"\nHighest dropout rate in the funnel: {highest_null_percentage:.2f}%")

# Find the difference between purchase_time and visit_time and create a new column to all_data
all_data['time_to_purchase'] = all_data['purchase_time'] - all_data['visit_time']
print("\nTime to purchase for each user:")
print(all_data['time_to_purchase'].head(5))

# Calculate the average time to purchase
average_time_to_purchase = all_data['time_to_purchase'].mean()
print(f"\nAverage time to complete purchase: {average_time_to_purchase}")
