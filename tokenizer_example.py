from transformers import AutoTokenizer

# Load the tokenizer for a specific pre-trained model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize a sentence
tokens = tokenizer.tokenize("Ananda Sathyaseelan")
print(tokens)
# ['i', 'love', 'pizza', '!']

# Convert tokens to IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)
# [1045, 2293, 10733, 999]
