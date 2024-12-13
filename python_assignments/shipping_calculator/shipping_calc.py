
# Calculating the cost of standard ground shipping
def ground_shipping_cost(weight):
# Under 2lb, shipping rate is $1.50 per lb plus $20 flat rate 
    if weight <= 2:
        cost = weight * 1.5 + 20
# Under 6lb but over 2lb, shipping rate is $3 per lb plus $20 flat rate
    elif weight <= 6:
        cost = weight * 3 + 20
# Under 10lb but over 6lb, shipping rate is $4 per lb plus $20 flat rate   
    elif weight <= 10:
        cost = weight * 4 + 20
# Anything over 10lb, shipping rate is $4.75 per lb plus $20 flat rate
    else:
        cost = weight * 4.75 + 20
    return cost
# If needed you can change these numbers to fit the company you are shipping with.

# Calculating the cost of Drone shipping, same as above just no flat rate
# and under 2lb is $4.50 per lb, over 2lb under 6lb is $9 per pound, over 6lb
# under 10lb is $12 per lb, and anything over 10lb is $14.25 per pound
def drone_shipping_cost(weight):
    if weight <= 2:
        cost = weight * 4.5
    elif weight <= 6:
        cost = weight * 9
    elif weight <= 10:
        cost = weight * 12
    else:
        cost = weight * 14.25
    return cost

# Calculating premuim ground shipping which is just a flatrate of $125 regardless
# of the weight.
def premium_ground_cost(weight):
    if weight >= 0:
        cost = 125
    return cost

# Comparing all the prices to give you the cheapest option.
def cheapest_shipping(weight):
    ground = ground_shipping_cost(weight)
    drone = drone_shipping_cost(weight)
    premium_ground = premium_ground_cost(weight)

    if ground < drone and ground < premium_ground:
        return "Ground Shipping", ground
    elif drone < ground and drone < premium_ground:
        return "Drone Shipping", drone
    else:
        return "Premium Ground Shipping", premium_ground

# Main function to print out all the results
def main():
    weight = float(input("Enter the weight of the package: "))
    method, cost = cheapest_shipping(weight)
    print("Cost of Ground Shipping: ${:.2f}".format(ground_shipping_cost(weight)))
    print("Cost of Premium Ground Shipping: ${:.2f}".format(premium_ground_cost(weight)))
    print("Cost of Drone Shipping: ${:.2f}".format(drone_shipping_cost(weight)))
    print("The cheapest method of shipping is {} and it will cost ${:.2f}".format(method, cost))

main()
