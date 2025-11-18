# Tiktoken library is used for implementing byte-pair encodings 

import tiktoken 

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    " of someunkownPlace."
)

integers = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
print("\nids= ",integers)
print("\n")

string = tokenizer.decode(integers)
print("text = ",string)
print("\n")

integers = tokenizer.encode("SAhvty jksd")
print ("ids 2 = ",integers)
print("\n")

string = tokenizer.decode(integers)
print("text 2 = ",string)
print("\n")

# Implementing Byte pair encoding on the book - the verdict

with open("the-verdict.txt", "r", encoding = 'utf-8') as f:
    raw_text = f.read()

print (f"\ntotal number of words = {len(raw_text)}\n")
print("original text = ",raw_text[:500])
print("\n")


enc_text = tokenizer.encode(raw_text)
print("length of encoded text = ",len(enc_text))
print("\n")

enc_sample = enc_text[50:]

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]

print ("x : ",x)
print ("y :     ",y)
print("\n")
