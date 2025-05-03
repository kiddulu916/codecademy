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

# Left merge visits and cart
visits_cart = pd.merge(visits, cart, how='left').reset_index()
print("\nMerged visits and cart data (first 5 rows):")
print(visits_cart.head(5))

# Find the length of visits_cart
length_of_visits_cart = visits_cart.size
print(f"\nTotal number of records in merged visits and cart: {length_of_visits_cart}")

# Find how many timestamps in the cart_time column are null
null_cart_time = visits_cart['cart_time'].isnull().size
print(f"Number of users who visited but didn't add to cart: {null_cart_time}")

# Calculate the percentages of null cart times
percentage_null_cart_time = (null_cart_time / length_of_visits_cart) * 100
print(f"Percentage of users who visited but didn't add to cart: {percentage_null_cart_time:.2f}%")

# Left merge cart and checkout
cart_checkout = pd.merge(cart, checkout, how='left').reset_index()
print("\nMerged cart and checkout data (first 5 rows):")
print(cart_checkout.head(5))

# Find the length of cart_checkout
length_of_cart_checkout = cart_checkout.size
print(f"\nTotal number of records in merged cart and checkout: {length_of_cart_checkout}")

# Find how many timestamps in the checkout_time column are null
null_cart_checkout_time = cart_checkout['checkout_time'].isnull().size
print(f"Number of users who added to cart but didn't proceed to checkout: {null_cart_checkout_time}")

# Calculate the percentages of null checkout times
percentage_null_cart_checkout_time = (null_cart_checkout_time / length_of_cart_checkout) * 100
print(f"Percentage of users who added to cart but didn't proceed to checkout: {percentage_null_cart_checkout_time:.2f}%")

# Left merge checkout and purchase
checkout_purchase = pd.merge(checkout, purchase, how='left').reset_index()
print("\nMerged checkout and purchase data (first 5 rows):")
print(checkout_purchase.head(5))

# Find the length of checkout_purchase
length_of_checkout_purchase = checkout_purchase.size
print(f"\nTotal number of records in merged checkout and purchase: {length_of_checkout_purchase}")

# Find how many timestamps in the purchase_time column are null
null_checkout_purchase_time = checkout_purchase['purchase_time'].isnull().size
print(f"Number of users who proceeded to checkout but didn't complete purchase: {null_checkout_purchase_time}")

# Calculate the percentages of null purchase times
percentage_null_checkout_purchase_time = (null_checkout_purchase_time / length_of_checkout_purchase) * 100
print(f"Percentage of users who proceeded to checkout but didn't complete purchase: {percentage_null_checkout_purchase_time:.2f}%")

# Merge all 4 dataframes
all_data = pd.merge(visits_cart, checkout_purchase, how='left').reset_index()
print("\nMerged data from all stages (first 5 rows):")
print(all_data.head(5))

# Find the length of all_data
length_of_all_data = all_data.size
print(f"\nTotal number of records in combined dataset: {length_of_all_data}")

# Find how many timestamps in the cart_time column are null
null_cart_time = all_data['cart_time'].isnull().size
print(f"Number of users who visited but didn't add to cart (from combined data): {null_cart_time}")

# Find how many timestamps in the checkout_time column are null
null_checkout_time = all_data['checkout_time'].isnull().size
print(f"Number of users who added to cart but didn't proceed to checkout (from combined data): {null_checkout_time}")

# Find how many timestamps in the purchase_time column are null
null_purchase_time = all_data['purchase_time'].isnull().size
print(f"Number of users who proceeded to checkout but didn't complete purchase (from combined data): {null_purchase_time}")

# Calculate the percentages of null cart times
percentage_null_cart_time = (null_cart_time / length_of_all_data) * 100
print(f"Percentage of users who visited but didn't add to cart (from combined data): {percentage_null_cart_time:.2f}%")

# Calculate the percentages of null checkout times
percentage_null_checkout_time = (null_checkout_time / length_of_all_data) * 100
print(f"Percentage of users who added to cart but didn't proceed to checkout (from combined data): {percentage_null_checkout_time:.2f}%")

# Calculate the percentages of null purchase times
percentage_null_purchase_time = (null_purchase_time / length_of_all_data) * 100
print(f"Percentage of users who proceeded to checkout but didn't complete purchase (from combined data): {percentage_null_purchase_time:.2f}%")

# Compare null percentages to find the highest percentage of users not completing a purchase
highest_null_percentage = max(percentage_null_cart_time, percentage_null_checkout_time, percentage_null_purchase_time)
print(f"\nHighest dropout rate in the funnel: {highest_null_percentage:.2f}%")

# Find the difference between purchase_time and visit_time and create a new column to all_data
all_data['time_to_purchase'] = all_data['purchase_time'] - all_data['visit_time']
print("\nTime to purchase for each user:")
print(all_data['time_to_purchase'])

# Calculate the average time to purchase
average_time_to_purchase = all_data['time_to_purchase'].mean()
print(f"\nAverage time to complete purchase: {average_time_to_purchase}")
