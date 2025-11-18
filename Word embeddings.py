import torch 
import torch.nn as nn
from collections import Counter
import tiktoken

# Loading dataset

with open("the-verdict.txt", "r", encoding = 'utf-8') as f:
    text = f.read()

print (f"\ntotal number of words = {len(text)}\n")
print("original text = ",text[:500])
print("\n")

# directly build token ids using byte pair encoding 

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text)

# Converting ids to tensors 

ids_to_tensors = torch.tensor(token_ids).unsqueeze(0)
# using squeeze to convert from one dimension to two, shape here will be [1, seq_len]
seq_len = ids_to_tensors.shape[1]
# number of tokens in the input 

# Define embedding layer

vocab_size = tokenizer.n_vocab # GPT 2 vocab size
embedding_dim = 16  
embedding_layer = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim)


# Get embeddings

embedded = embedding_layer(ids_to_tensors)
print("\nEmbeddings shape:", embedded.shape)
print("Sample embedding for first token:", embedded[0])


