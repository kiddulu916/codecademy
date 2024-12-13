import string

def encrypt(text, shift):
    encrypted_text = list(range(len(text)))
    alphabet = string.ascii_lowercase # 'abcdefghijklmnopqrstuvwxyz'
    first_half = alphabet[:shift]
    second_half = alphabet[shift:]
    shifted_alphabet = second_half + first_half
    
    for i, letter in enumerate(text.lower()):

        if letter in alphabet:
            original_index = alphabet.index(letter)
            new_letter = shifted_alphabet[original_index]
            encrypted_text[i] = new_letter    
        else:
            encrypted_text[i] = letter

    return "".join(encrypted_text)

text = input("Enter text to encrypt: ")
shift = int(input("Enter the shift value: "))
encrypted = encrypt(text, shift)
print(f"Encrypted message is: {encrypted}")
