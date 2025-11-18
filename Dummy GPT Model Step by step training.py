# Step 1 - Creating GPT placeholder 
# Step 2 - Testing Layer Normalization 
# Step 3 - GeLU activation Function 
# Step 4 - Feedforwards network
# Step 5 - Skip Connections 

import torch
import torch.nn as nn

# Step 1 
# Dummy GPT placeholder 

GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "emb_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qvk_bias" : False
}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Use a placeholder for transformer block
        self.trf_blocks  = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range (cfg["n_layers"])]
        )

        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias = False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device = in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing, just returns its input 
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.

    def forward(self, x):
        # This layer does nothing and just returns its input.
        return x
  
# Tokenization

import tiktoken
from torch.nn.utils.rnn import pad_sequence

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
encoded1 = torch.tensor(tokenizer.encode(txt1))
encoded2 = torch.tensor(tokenizer.encode(txt2))

batch = pad_sequence([encoded1, encoded2], batch_first=True, padding_value=0)
print("\nbatch = \n",batch)

# Creating instance for DummyGPTModel

print("\n=========== Testing Dummy GPT model ===========")

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("\nOutput shape = ", logits.shape)
print("\nOutput = ", logits)


# Step 2 
# Layer Normalization 
 
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        # eps is small constant epsilon added to prevent division by zero 

        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        return self.scale * norm_x + self.shift

        # scale and shifts are two parameters (of the same dimension as input)
        # LLM automatically adjust during training to improve model's performance
        # This allows the model to learn appropriate scaling and shifting that best suit the data in processing

# Creating temproary batch for testing layer normalization 

torch.manual_seed(123)
batch_example = torch.randn(2, 5)   # input - 2 rows, 5 columns
layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())    # 5 inputs, 6 outputs
out = layer(batch_example)
print("\noutput of layer norm = ", out)

print("\n=========== Testing Layer Normalization ===========")

ln = LayerNorm(emb_dim = 5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim = -1, keepdim = True)
var = out_ln.var(dim = -1, keepdim = True)

print("\nNormalized output = ", out_ln)
print("\nMean after normalizarion = ", mean)
print("\nVariance after normalization = ", var)

# Step 3 
# GELU Activation function 

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044714 * torch.pow(x, 3))
        ))

# Visualizing GELU activation function 

import matplotlib.pyplot as plt

gelu, relu = GELU(), nn.ReLU()

# Some sample data
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

print("\n=========== Testing GELU activation function against ReLU ===========")

# plt.figure(figsize = (12,8))

# for i, (y, lable) in enumerate (zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
#     plt.subplot(1,2,i)
#     plt.plot(x,y)
#     plt.title(f"{lable} Activation function")
#     plt.xlabel("x")
#     plt.ylabel(f"{lable} x")
#     plt.grid(True)

# plt.tight_layout()
# plt.show()

# Step 4
# Feedforward network - it will use GELU activation function 

class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # Expansion
            GELU(),     # Activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])   # Contraction 
        )
    
    def forward(self, x):
        return self.layers(x)
    
print("\nInput embedding dimension = ",GPT_CONFIG_124M["emb_dim"])

print("\n=========== Testing Feedforward network ===========")

# Instance of feedforward network class 

ffn = FeedForwardNetwork(GPT_CONFIG_124M)
x = torch.rand(2,3,768) 
out = ffn(x)

print("\nOutput embedding Dimension = ", out.shape)

# Step 5 
# Skip connections 

print("\n=========== Testing Skip Connections ===========")

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_size, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_size[0], layer_size[1], GELU())),
            nn.Sequential(nn.Linear(layer_size[1], layer_size[2], GELU())),
            nn.Sequential(nn.Linear(layer_size[2], layer_size[3], GELU())),
            nn.Sequential(nn.Linear(layer_size[3], layer_size[4], GELU())),
            nn.Sequential(nn.Linear(layer_size[4], layer_size[5], GELU()))
        ])

    def forward(self, x):
        for layer in self.layers:
            # Computing the output of current layer 
            layer_output = layer(x)

            # Check if shortcut can be applied 
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x
    
layer_size = [3,3,3,3,3,1]     # tells neurons per layer
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_size, use_shortcut = False)

def Print_Gradients(model,x):
    # Forward pass 
    output = model(x)
    target = torch.tensor([[0.]])

    # Caluclating loss on basis of how close target are to outputs
    loss = nn.MSELoss() 
    loss = loss(output, target)

    # Backword pass to calculate gradient
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights 
            print(f"{name}  has gradient mean of {param.grad.abs().mean().item()}")

print("\nPrinting Gradients for model WITHOUT shortcut connections : \n")
Print_Gradients(model_without_shortcut, sample_input)

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_size, use_shortcut = True)

print("\nPrinting Gradients for model WITH shortcut connections : \n")
Print_Gradients(model_with_shortcut, sample_input)




