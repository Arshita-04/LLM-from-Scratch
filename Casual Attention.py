import torch
import torch.nn as nn
import torch.functional as F

with open ("the-verdict.txt", "r", encoding = "utf-8") as f:
    text = f.read().strip().lower()

print("\ntotal length of text = ", len(text))
print("\nOriginal text = ", text[:500])

# Tokenize 

tokens = text.split()
all_words = sorted(set(tokens))
vocab = {token:integer for integer,token in enumerate(all_words)}
int_vocab = {integer:token for token,integer in vocab.items()}
token_ids = [vocab[word] for word in tokens]
 
# Convert to tensors

x = torch.tensor(token_ids).unsqueeze(0)

# Hyperparameters

embedding_dim = 32
vocab_size = len(vocab)
seq_len = x.size(1)

embedding = nn.Embedding(vocab_size, embedding_dim)
query_vec = nn.Linear(embedding_dim, embedding_dim, bias = False)
key_vec = nn.Linear(embedding_dim, embedding_dim, bias = False)
value_vec = nn.Linear(embedding_dim, embedding_dim, bias = False)

# Defining Casual attention 

def CasualAttention(x_embedd):
    
    # Computes causal self-attention on the input embeddings.
    # x_embed: Tensor of shape [batch_size, seq_len, embed_dim]

    batch, time, channels = x_embedd.shape

    Q = query_vec(x_embedd)
    K = key_vec(x_embedd)
    V = value_vec(x_embedd)

    attention_score = torch.matmul(Q, K.transpose(-2, -1)) / (embedding_dim ** 0.5)

    # Create upper triangular casual mask 

    mask = torch.triu(torch.ones(time, time), diagonal = 1).bool().to(x_embedd.device)
    attention_score = attention_score.masked_fill(mask, float('-inf'))

    # Apply softmax to get the weights 
    attention_weight = torch.softmax(attention_score, dim = 1)

    # Multiply weights with value 
    output = torch.matmul(attention_weight, V)

    return output, attention_weight

# Forward pass through embedding layer

x_embed = embedding(x)

# Apply casual attention

attention_output, attention_weights = CasualAttention(x_embed) 

# Print sample results

print("\nTokens : \n", tokens[:10])
print("\nToken IDs : \n", token_ids[:10])
print("\nAttention Weights for first 10 tokens :\n", attention_weights[0, :10, :10].detach())

