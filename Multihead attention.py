import torch
import torch.nn as nn

with open("the-verdict.txt", "r", encoding = 'utf-8') as f:
    text = f.read()

print("\nthe length of the text = ", len(text))
print("\nsample text = ", text[:500])

# Tokenize

tokens = text.split()
all_words = sorted(set(tokens))
vocab = {token:integer for integer,token in enumerate(all_words)}
int_vocab = {integer:token for token, integer in vocab.items()}
token_ids = [vocab[word] for word in tokens]

# converting to tensors 

x = torch.tensor(token_ids).unsqueeze(0)

# hyperparameters

embedding_dim = 32
vocab_size = len(all_words)
seq_len = x.size(1)

# getting embeddings

embedding = nn.Embedding(vocab_size, embedding_dim)

class MultiheadAttention (nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads      # Reducing the projection dimension to match desired output dimension

        # Initializing weight matrices for key, query and value
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

        self.out_proj = nn.Linear(d_in, d_out)  # Linear layer to combine head output 
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask (upper triangular matrix)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # b - batch, input = num_tokens, d_in = input dimension
        # d_out = output dimension

        # Calculating the three vectors
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        atten_score = queries @ keys.transpose(2,3)
        
        mask_bool = self.mask [:num_tokens, :num_tokens]

        atten_score.masked_fill(mask_bool, -torch.inf)

        atten_weights = torch.softmax(atten_score / keys.shape [-1] ** 0.5, dim = -1)
        atten_weights = self.dropout(atten_weights)

        context_vec = (atten_weights @ values).transpose(1,2)

        # Combine heads
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec, atten_weights
    

x_embed = embedding(x)

# Apply attention
attention = MultiheadAttention(
    d_in = embedding_dim,
    d_out = embedding_dim,
    context_length = seq_len,
    dropout = 0.1,          
    num_heads = 4
)

output, attn_weights = attention(x_embed)

# Sample outputs
print("\nTokens : \n", tokens[:10])
print("\nToken IDs : \n", token_ids[:10])
print("\nAttention Weights for first 10 tokens :\n", attn_weights[0, 0, :10, :10].detach())
