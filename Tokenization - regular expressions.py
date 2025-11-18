with open("the-verdict.txt","r", encoding = 'utf-8') as f: 
    raw_text = f.read()         # utf-8 --> Unicode Transformation Format - 8-bit, default encoding for strings in python 

print (f"\ntotal number of words = {len(raw_text)}\n")
print("original text = ",raw_text[:500])
print("\n")

# Tokenization
# Converting the entire story The Verdict in tokens

import re

preprocessed = re.split(r'([,.:;?"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("preprcessed text = ",preprocessed[:90])

print("length of preprocessed text = ",len(preprocessed))
print("\n")

#Token-Ids
#Now converting all tokens in to tokenIds (vectors) means the words are now in numerical representation 

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print("vocab size = ",vocab_size)
print("\n")

# Vocabulary
# A vocabulary is like a dictionary, with tokens assigned to their tokenIds or vectors

vocab = {token:integer for integer,token in enumerate(all_words)}
print("printing vocabulary : \n")

for i, item in enumerate(vocab.items()):
    print(item)
    if i>10:
        break
print("\n")

# Encoding and decoding the above tokens using word-tokenization

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        
        preprocessed = re.split(r'([,.:;?"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
       
        text = " ".join([self.int_to_str[i] for i in ids])
        
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

        return text


tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride."""

ids = tokenizer.encode(text)
print("ids = ",ids)
print("\n")

string = tokenizer.decode(ids)
print("text = ",string)
print("\n")

# text = "hello, do you like tea?"
# print(tokenizer.encode(text))

# hello is the new word here and not present in vocabulary hence it gave error

# Adding Special Context tokens
# so the tokenizer can identify handle unknown words

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}

print("vocab size = ",len(vocab.items()))
print("\n")

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        
        preprocessed = re.split(r'([,.:;?"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int
                       else "<|unk|>" for item in preprocessed]
        
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
       
        text = " ".join([self.int_to_str[i] for i in ids])
        
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

        return text

tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "Mrs. Gisburn said with pardonable pride."

text = " <|endoftext|> ".join((text1, text2))

ids = tokenizer.encode(text)
print("ids = ",ids)
print("\n")

print("text = ",tokenizer.decode(tokenizer.encode(text)))
print("\n")



