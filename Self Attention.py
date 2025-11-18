import tiktoken
import torch
import torch.nn as nn

with open("the-verdict.txt", "r", encoding = "utf-8") as f:
    text = f.read()

print("\nLength of text = ", len(text))
print("\nOriginal text = ", text[:500])

# Tokenize

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text)
token_tensors = torch.tensor(token_ids).unsqueeze(0)
seq_len = token_tensors.shape[1]

# Token Embeddings 

embedding_dim = 32
vocab_size = tokenizer.n_vocab
embedding = nn.Embedding(vocab_size, embedding_dim)
embedded = embedding(token_tensors)

# Self attention Module

class SelfAttention(nn.Module):
    def __init__(self, embedd_dim):
        
        super().__init__()
        
        # passing each token embedding through these three vectors 

        self.query = nn.Linear(embedd_dim, embedd_dim)
        self.key = nn.Linear(embedd_dim, embedd_dim)
        self.value = nn.Linear(embedd_dim, embedd_dim)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self,x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Compute attention score

        score = torch.matmul(Q, K.transpose(-2,-1)) / (x.size(-1) ** 0.5)
        
        # Compute weights 

        weights = self.softmax(score)

        # Weighted sum of values 

        output = torch.matmul(weights, V)

        return output, weights

# Apply attention

attention = SelfAttention(embedding_dim)
attention_output, attention_weights = attention(embedded)

# Outputs 

print("\nInput embedding shape:", embedded.shape)
print("\nInput embedding :", embedded)

print("\nAttention output shape:", attention_output.shape)
print("\nAttention output :", attention_output)

print("\nAttention weights shape:", attention_weights.shape)
print("\nAttention weights :", attention_weights)
print("\n")

