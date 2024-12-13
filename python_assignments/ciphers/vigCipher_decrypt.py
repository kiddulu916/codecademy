def decrypt(ciphertext, key):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    decrypted_text = ""
    key_index = 0

    for char in ciphertext:
        if char.isalpha():
            # Find the position in the alphabet
            char_index = alphabet.find(char.lower())
            # Find the position of the key character
            key_char = key[key_index % len(key)]
            key_char_index = alphabet.find(key_char)
            # Decrypt the character
            decrypted_char = alphabet[(char_index + key_char_index) % len(alphabet)]
            # Append the decrypted character to the decrypted text
            decrypted_text += decrypted_char
            # Only increment key_index if a letter was encrypted/decrypted
            if char.isalpha():
                key_index += 1
        else:
            # If the character is not a letter, add it as it is (preserving spaces)
            decrypted_text += char

    return decrypted_text

# Encrypted message and key
encrypted_message = input("Enter the encrypted message: ")
key = input("Enter the key: ")

# Repeat the key to match the length of the encrypted message
repeated_key = (key * (len(encrypted_message) // len(key))) + key[:len(encrypted_message) % len(key)]

# Decrypt the message using the repeated key
decrypted_message = decrypt(encrypted_message, repeated_key)

print(f"Decrypted message: {decrypted_message}")