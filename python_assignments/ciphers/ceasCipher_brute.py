# Ask the user for the encrypted message
message = input("Enter encrypted message to decrypt: ")
# Ask user for the shift value
offset = int(input("Enter the shift value: "))

# Function for decryption
def decrypt(message, offset):
    # Stores the letters of the alphabet
    letters = 'abcdefghijklmnopqrstuvwxyz'
    # Empty string variable to catch the deciphered text
    cipher = ''
    
    # Loop through the characters of the encrypted message
    for i in message:
        # checks if each character of the encrypted message matches a character in letters  
        if i in letters:
            # Deciphers each character of the encrypted message and adds each deciphered character to the cipher variable
            cipher += letters[(letters.index(i) + offset)%(len(letters))]
        # Adds characters in the encrypted message that are not characters in the letters variable 
        else:
             cipher += i
    # Returns the deciphered message
    return cipher

# Calls the decrypt function with variables message and shift as arguments and stores the output to the variable decrypted
decrypted = decrypt(message, offset)        

# Prints decrypted message 
print(f"Decrypted message is: {decrypted}")

def brute_force_decrypt(message):
    #when the shift is unknown
    for n in range(26):
        print(f"Using a shift value of {n}")
        print(decrypt(message, n))
        print("\n***\n")

brute = input("Enter encrypted message to brute force: ") 
brute = brute_force_decrypt(brute)