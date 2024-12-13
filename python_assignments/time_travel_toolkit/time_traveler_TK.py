import datetime as dt
from decimal import Decimal
from random import randint, choice
import custom_module as cm

# Grab the current year and current time.
current_year = dt.date.today().year
time_now = dt.datetime.now().time()

# Printing out the curent year and time.
print(f"Todays year is {current_year}\nTime: {time_now}")

# Generate a Decimal object for base travel cost and base year of travel.
base_cost = Decimal("10.75")
base_year = Decimal(current_year)

# Generate a random year to travel to.
random_year = randint(0, 99999)

# Calculate the cost multiplier.
cost_multiplier = abs(random_year - base_year)

# Calculate the travel cost.
travel_cost = base_cost * cost_multiplier

# round the travel cost to 2 decimal places.
final_travel_cost = round(travel_cost, 2)

# Create a list of destinations.
destinations = ["Italy", "France", "USA", "Peru", "Egypt"]

# Variable that will choose a random destination.
random_dest = choice(destinations)

# Generate the time travel message.
cm.generate_time_travel_message(random_year, random_dest, final_travel_cost)
