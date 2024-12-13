import enchant

# Letters and points to map together
letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
points = [1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 4, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10]

# Create an English dictionary to check words that were entered
dict = enchant.Dict("en_US")

# Create a dictionary mapping letters to point values
letter_to_points = {letter: point for letter, point in zip(letters, points)}
letter_to_points[" "] = 0

#Empty list to hold players and there scores for each player
players = []

# Empty list to seperate the players and scores to there own lists
player1 = []
player2 = []
player3 = []
player4 = []

# Function add the players to the players list
def add_players():
    print("\n\nOnly up to 4 players. If there is less then 4 leave additional players blank and press enter.\n\n")
    # Starts while loop asking for players names
    while len(players) < 4:
        new_player = input("Enter player to add>>> ")
        # Checks if the player is already in the players list
        if new_player:
            if new_player not in (stats[0] for stats in players):
                players.append([new_player, 0])
                print("Player {} added.\n\n".format(new_player))
            else:
                print("Player already in player list.\n\n")
        else:
            break

# Function to calculate the score of the words entered
def word_score(word):
    total = 0
    # Loop through each letter in the word and add the point value to the total
    for letter in word:
        return sum(letter_to_points[letter] for letter in word)

# Function to ask if there are more turns need to be played
def rounds():
    # If no more rounds then calculate the scores
    if input("\n\nIs there another round? (y/n): ") == "n":
        print("\n\nCalculating scores...\n")
        total_score()
    # If there are more rounds then restart the turns function
    else:
        print("\n\nNext round....\n")
        turns()
        
# Function to cycle through the players turns
def turns():
    # states that its player 1's turn
    print(f"\nIt\'s {player1[0]}\'s turn.")
    # Variable to add the score for player 1
    score1 = 0
    # Loop to start player 1's turn again if given invalid input
    while True:
        # Ask for a word from player 1
        word1 = input(f"\n{player1[0]}, please enter a word: ")
        # Check if the word is in the dictionary if not then asks for another word
        if dict.check(word1) == False:
            print("That's not a valid word. Please try again.\n\n")
        # If the word is in the dictionary then add the score to the total
        elif dict.check(word1) == True:
            # Add the score to the total using the word_score function
            score1 += word_score(word1)
            # Add the score to player 1's score list
            player1[1] += score1
            # Prints player 1's score
            print(f"{player1[0]}, your score is {score1}.\n\n")
            # Break out of the while loop
            break
        # If input request is left blank then ask for another word
        else:
            print("You cant leave this blank. Please try again.\n\n")
            
    # States that its player 2's turn
    print(f"\nIt\'s {player2[0]}\'s turn.")
    # Variable to add the score for player 2
    score2 = 0
    # Loop to start player 2's turn
    while True:
        # Ask for a word from player 2
        word2 = input(f"\n{player2[0]}, please enter a word: ")
        # Check if the word is in the dictionary if not then asks for another word
        if dict.check(word2) == False:
            print("That's not a valid word. Please try again.\n\n")
        # If the word is in the dictionary then add the score to the total
        elif dict.check(word2) == True:
            # Add the score to the total using the word_score function
            score2 += word_score(word2)
            # Add the score to player 2's score list
            player2[1] += score2
            # Prints player 2's score
            print(f"\n{player2[0]}, your score is {score2}.\n\n")
            # Break out of the while loop
            break
        # If input request is left blank then ask for another word
        else:
            print("You cant leave this blank. Please try again.\n\n")
            
        # If player 3's list is not empty then its player 3's turn
        if player3 != []:
            # States that its player 3's turn
            print(f"\nIt\'s{player3[0]}\'s turn.")
            # Variable to add the score for player 3
            score3 = 0
            # Loop to start player 3's turn if given invalid input
            while True:
                # Ask for a word from player 3
                word3 = input(f"\n{player3[0]}, please enter a word: ")
                # Check if the word is in the dictionary if not then asks for another word
                if dict.check(word3) == False:
                    print("That's not a valid word. Please try again.\n\n")
                # If the word is in the dictionary then add the score to the total
                elif dict.check(word3) == True:
                    # Add the score to the total using the word_score function
                    score3 += word_score(word3)
                    # Add the score to player 3's score list
                    player3[1] += score3
                    # Prints player 3's score
                    print(f"\n{player3[0]}, your score is {score3}.\n\n")
                    # Break out of the while loop
                    break
                # If input request is left blank then ask for another word
                else:
                    print("You cant leave this blank. Please try again.\n\n")
         
        # If player 4's list is not empty then its player 4's turn
        if player4 != []:
            # States that its player 4's turn
            print(f"\nIt\'s {player4[0]}\'s turn.")
            # Variable to add the score for player 4
            score4 = 0
            # Loop to start player 4's turn if given invalid input
            while True:
                # Ask for a word from player 4
                word4 = input(f"{player4[0]}, please enter a word: ")
                # Check if the word is in the dictionary if not then asks for another word
                if dict.check(word4) == False:
                    print("That's not a valid word. Please try again.")
                # If the word is in the dictionary then add the score to the total
                elif dict.check(word4) == True:
                    # Add the score to the total using the word_score function
                    score4 += word_score(word4)
                    # Add the score to player 4's score list
                    player4[1] += score4
                    # Prints player 4's score
                    print(f"{player4[0]}, your score is {score4}.\n\n")
                    # Break out of the while loop
                    break
                # If input request is left blank then ask for another word
                else:
                    print("You cant leave this blank. Please try again.")
    
    # Starts the rounds function                
    rounds()

# Function to calculate the total score of the players
# *Side Note* I compared the scores in this manner 1) because im not very great with
# while and for loops yet and 2) to avoid having any errors if there was no player 3 
# or player 4 given. I know I couldve done this with try/except block but found out
# I can't place a try/except block in the middle of an if statement and wrapping the
# if statement in a try/except block still gave me errors, so I decided to just write 
# it all out in elif statements. If anyone knows a better way to do this then please
# share, I always love to learn better ways to code. 
def total_score():
    # Prints the total score of player 1
    print(f"{player1[0]}'s score: {player1[1]}")
    # Prints the total score of player 2
    print(f"{player2[0]}'s score: {player2[1]}")
    # If player 3's list is not empty then prints the total score of player 3
    if player3 != []:
        print(f"{player3[0]}'s score: {player3[1]}")
    # If player 4's list is not empty then prints the total score of player
    if player4 != []:
        print(f"{player4[0]}'s score: {player4[1]}")
    # Compares player 1's score to player 2's score
    if player1[1] > player2[1]:
        print(f"\n\n{player1[0]} is the winner!")
    # Compares player 1's score to player 2's and player 3's score 
    elif player1[1] > player2[1] and player1[1] > player3[1]:
        print(f"\n\n{player1[0]} is the winner!")
    # Compares player 1's score to player 2's, player 3's and player 4's score
    elif player1[1] > player2[1] and player1[1] >player3[1] and player1[1] > player4[1]:
        print(f"\n\n{player1[0]} is the winner!")
    # Compares player 2's score to player 1's score
    elif player2[1] > player1[1]:
        print(f"\n\n{player2[0]} is the winner!")
    # Compares player 2's score to player 1's and player 3's score
    elif player2[1] > player1[1] and player2[1] > player3[1]:
        print(f"\n\n{player2[0]} is the winner!")
    # Compares player 2's score to player 1's, player 3's and player 4's score
    elif player2[1] > player1[1] and player2[1] > player3[1] and player2[1] > player4[1]:
        print(f"\n\n{player2[0]} is the winner!")
    # Compares player 3's score to player 1's and player 2's
    elif player3[1] > player1[1] and player3[1] > player2[1]:
        print(f"\n\n{player3[0]} is the winner!")
    # Compares player 3's score to player 1's, player 2's and player 4's score
    elif player3[1] > player1[1] and player3[1] > player2[1] and player3[1] > player4[1]:
        print(f"\n\n{player3[0]} is the winner!")
    # Compares player 4's score to player 1's, player 2's and player 4's score
    elif player4[1] > player1[1] and player4[1] > player2[1] and player4[1] > player3[1]:
        print(f"\n\n{player4[0]} is the winner!")
    # Otherwise prints that its a tie
    else:
        print("\n\nIt's a tie!")

# Main function to start the game        
def main():
    #calls the add_players function
    add_players()
    # If statement if only 2 players are given
    if len(players) == 2:
        # Separate the lists in the players list to there own lists for player 1 and 2
        player1.append(players[0][0])
        player1.append(players[0][1])
        player2.append(players[1][0])
        player2.append(players[1][1])
        print(f"\n\n{player1[0]} and {player2[0]} Let's play Scrabble!\n\n")
    # Else/If statement if only 3 players are given
    elif len(players) == 3:
        # Separate the lists in the players list to there own lists for player 1, 2, and 3
        player1.append(players[0][0])
        player1.append(players[0][1])
        player2.append(players[1][0])
        player2.append(players[1][1])
        player3.append(players[2][0])
        player3.append(players[2][1])
        print(f"\n\n{player1[0]}, {player2[0]}, and {player3[0]} Let's play Scrabble!\n\n")
    # Else/If statement if only 4 players are given
    elif len(players) == 4:
        # Separate the lists in the players list to there own lists for player 1, 2, 3, and 4
        player1.append(players[0][0])
        player1.append(players[0][1])
        player2.append(players[1][0])
        player2.append(players[1][1])
        player3.append(players[2][0])
        player3.append(players[2][1])
        player4.append(players[3][0])
        player4.append(players[3][1])
        print(f"\n\n{player1[0]}, {player2[0]}, {player3[0]}, and {player4[0]} Let's play Scrabble!\n\n")
    # In case someone found a way to make more then 4 players it'll catch the error and repeat add_players function 
    else:
        print("\n\nInvalid number of players. Please try again.\n\n")
        add_players()
    # Starts the turns function
    turns()
# Calls the main function to start the game
main()