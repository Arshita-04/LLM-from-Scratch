import torch
import torch.nn as nn
from collections import Counter
import tiktoken

with open("the-verdict.txt", "r", encoding = "utf-8") as f:
    text = f.read()

print (f"\ntotal number of words = {len(text)}\n")
print("original text = ",text[:500])
print("\n")

# directly build token ids using byte pair encoding 

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text)

# convert to tensors 

ids_to_tensor = torch.tensor(token_ids).unsqueeze(0)  
# using squeeze to convert from one dimension to two, shape here will be [1, seq_len]
seq_len = ids_to_tensor.shape[1]
print("seq_len = ", seq_len)
# number of tokens in the input 

# token embeddings (Learned)

vocab_size = tokenizer.n_vocab
embedding_dim = 32
token_embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim)

print("\ntoken_embedding = ", token_embedding)
# this is creating token embedding layer

# Positional Embeddings (Learned)

max_seq_len = seq_len
positional_embeddings = nn.Embedding(num_embeddings = max_seq_len, embedding_dim = embedding_dim)

print("\npositional_embeddings = ", positional_embeddings)
# this is creating positional embedding layer 

# Generate positional ids 

position_ids = torch.arange(seq_len).unsqueeze(0)
print("\nposition_ids = ", position_ids)

# Get embeddings 

token_emb = token_embedding(ids_to_tensor)

position_emb = positional_embeddings(position_ids)

final_emb = token_emb + position_emb

# Outputs

print("\nToken Embedding shape:", token_emb.shape)
print("\nToken Embedding :", token_emb)

print("\nPositional Embedding shape:", position_emb.shape)
print("\nPositional Embedding :", position_emb)

print("\nFinal Combined Embedding shape:", final_emb.shape)
print("\nFinal Combined Embedding :", final_emb)

print("\nSample Combined Embedding for First Token:\n", final_emb[0, 0])

