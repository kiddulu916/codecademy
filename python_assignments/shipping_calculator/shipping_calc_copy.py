
# Calculating the cost of standard ground shipping
def ground_shipping_cost(weight, min_weight, max_weight, mid_weight, min_rate, mid_rate, mid_rate2, max_rate, flat_rate):
    if weight <= min_weight:
        cost = weight * min_rate + flat_rate
    elif weight <= mid_weight:
        cost = weight * mid_rate + flat_rate  
    elif weight <= max_weight:
        cost = weight * mid_rate2 + flat_rate
    else:
        cost = weight * max_rate + flat_rate
    return cost

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
    min_weight = float(input("Enter the minimum weight for the shipping rates: "))
    mid_weight = float(input("Enter the middle weight for the shipping rates(min-mid): "))
    max_weight = float(input("Enter the max weight for the shipping rates(mid-max): "))
    min_rate = float(input("Enter the minimum cost per lb for shipping: "))
    mid_rate = float(input("Enter the lower medium cost per lb for shipping: "))
    mid_rate2 = float(input("Enter the higher medium cost per lb for shipping: "))
    max_rate = float(input("Enter the maximum cost per lb for shipping: "))
    weight = float(input("Enter the weight of the package: "))
    method, cost = cheapest_shipping(weight)
    print("Cost of Ground Shipping: ${:.2f}".format(ground_shipping_cost(weight)))
    print("Cost of Premium Ground Shipping: ${:.2f}".format(premium_ground_cost(weight)))
    print("Cost of Drone Shipping: ${:.2f}".format(drone_shipping_cost(weight)))
    print("The cheapest method of shipping is {} and it will cost ${:.2f}".format(method, cost))

if __name__ == "__main__":
    main()
