# Components of Transformer block
# 1. Multihead attention
# 2. Layer Normalization
# 3. Dropout
# 4. Feedforward Layers
# 5. GELU Activation Function

import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "emb_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}

# Multihead attention 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisble by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal = 1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        query = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        query = query.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1,2)
        query = query.transpose(1,2)
        values = values.transpose(1,2)

        # Compute self attention with a casual mask
        att_score = query @ keys.transpose(2,3) # Dot product for each head

        mask_bool = self.mask[:num_tokens, :num_tokens].bool()

        att_score.masked_fill(mask_bool, -torch.inf)

        atten_weights = torch.softmax(att_score / keys.shape [-1] ** 0.5, dim = -1)
        atten_weights = self.dropout(atten_weights)

        context_vec = (atten_weights @ values).transpose(1,2)

        # Combine heads
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

# Layer Normalization 

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5

        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        return self.scale * norm_x + self.shift

# GELU Activation function 

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044714 * torch.pow(x, 3))
        ))
    
# Feedforward Network

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # Expansion
            GELU(),     # Activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])   # Contraction 
        )
    
    def forward(self, x):
        return self.layers(x)
     

# Transformer block 

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],          
            qkv_bias = cfg["qkv_bias"]
        )
        
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)         # Shape - [batch_size, num_tokens, emb_dim]
        x = self.dropout_shortcut(x)
        x = x + shortcut        # Add the original input back

        # Shortcut connection for feedforward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut        # Add the original input back

        return x
    
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("\nInput shape = ", x.shape)
print("\nInput = \n", x)
print("\nOutput shape = ", output.shape)
print("\nOutput = \n", output)
