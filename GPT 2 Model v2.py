import torch 
import torch.nn as nn
import json

# Parameters  

GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "emb_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}

# MultiHead Attention 

class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisble by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

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

        # Compute self attention using casual mask 
        att_score = query @ keys.transpose(2,3)

        mask_bool = self.mask[:num_tokens, :num_tokens].bool()

        att_score.masked_fill(mask_bool, -torch.inf)

        atten_weight = torch.softmax(att_score / keys.shape[-1] ** 0.5, dim = -1)
        atten_weight = self.dropout(atten_weight)

        context_vec = (atten_weight @ values).transpose(1,2)

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
        var = x.var(dim = -1, keepdim = True)
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

# Feedforwad Network

class Feedforward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Calls GELU function 
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)

# Transformer Block

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Calls MultiheadAttention function 
        self.att = MultiheadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],          
            qkv_bias = cfg["qkv_bias"]
        )

        # Calls Feedforward Function and LayerNorm function 
        self.ff = Feedforward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        attn_shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout_shortcut(x)
        x = x + attn_shortcut

        # Shortcut connection for feedforward block
        ff_shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout_shortcut(x)
        x = x + ff_shortcut

        return x
    
# GPT Model 

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_block = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias = False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape 
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device = in_idx.device)).unsqueeze(0)   

        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_block(x)       # Calls Transformer block 
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits
    

# Imports

import tiktoken
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset

# Hyper parameters 

block_size = 128
batch_size = 16
epochs = 50
lr = 1e-4   # Learning rate
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load dataset and Tokenize

def LoadTokenizedData(file_path):
    with open(file_path, "r", encoding = "utf-8") as f:
        text = f.read()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer, tokenizer.encode(text)


# GPT dataset loading

class GPTDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size
    
    # This is creating input-output pairs 
    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.block_size], dtype = torch.long)
        y = torch.tensor(self.tokens[idx + 1 : idx + self.block_size + 1], dtype = torch.long)

        return x, y

# Training function 

def train(model, dataloader, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")

# Main Script 
if __name__ == "__main__":
    tokenizer, tokens = LoadTokenizedData("file.txt")
    dataset = GPTDataset(tokens, block_size)
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

    model = GPTModel(GPT_CONFIG_124M).to(device)
    optimizer = optim.AdamW(model.parameters(), lr = lr)

    train(model, dataloader, optimizer, device, epochs)

def generate_answer(prompt, max_tokens = 100):
    # Encode input
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # Generate output tokens
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_tensor)
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_token_id]], device=device)], dim=1
            )
            if next_token_id == tokenizer.encode("\n")[0]:
                break

    output_tokens = input_tensor[0].tolist()
    output_text = tokenizer.decode(output_tokens)
    return output_text[len(prompt):].strip()

# Interactive QA loop
while True:
    user_input = input("\nAsk a question (type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break

    prompt = f"Question: {user_input}\nAnswer:"
    answer = generate_answer(prompt)
    print(f"Answer: {answer}")



