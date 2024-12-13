def vigenere_encrypt(plaintext, key):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    encrypted_text = ""
    key_index = 0

    for char in plaintext:
        if char.isalpha():
            # Find the position in the alphabet
            char_index = alphabet.find(char.lower())
            # Find the position of the key character
            key_char = key[key_index % len(key)]
            key_char_index = alphabet.find(key_char)
            # Encrypt the character
            encrypted_char = alphabet[(char_index - key_char_index) % len(alphabet)]
            # Append the encrypted character to the encrypted text
            encrypted_text += encrypted_char
            # Only increment key_index if a letter was encrypted/decrypted
            if char.isalpha():
                key_index += 1
        else:
            # If the character is not a letter, add it as it is (preserving spaces or special characters)
            encrypted_text += char

    return encrypted_text

# Plain message and key
plain_message = input("Enter the message for encryption: ")
key = input("Enter the key: ")

# Repeat the key to match the length of the plain message
repeated_key = (key * (len(plain_message) // len(key))) + key[:len(plain_message) % len(key)]

# Encrypt the message using the repeated key
encrypted_message = vigenere_encrypt(plain_message, repeated_key)

print(f"Encrypted message: {encrypted_message}")