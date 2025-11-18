import torch 
import torch.nn as nn

torch.manual_seed(123)
batch_example = torch.randn(2, 5)   # input - 2 rows, 5 columns
layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())    # 5 inputs, 6 outputs
out = layer(batch_example)
print("\noutput of layer norm = ", out)

mean = out.mean(dim = -1, keepdim = True)
var = out.var(dim = -1, keepdim = True)
print("\nMean = ", mean)
print("\nVariance = ", var)

# Normalization means the mean of the inputs should be zero and variance should be one

# Normalization formula is :
# calculate mean and varince for all inputs 
# normalization = (inputs-means)/sqrt(var)

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim = -1, keepdim = True)
var = out_norm.var(dim = -1, keepdim = True)

print("\nNormalized output = ", out_norm)
print("\nMean after normalizarion = ", mean)
print("\nVariance after normalization = ", var)

print("\nValues without scientific notions : ")

torch.set_printoptions(sci_mode = False)
print("\nMean = ", mean) 
print("\nVariance = ", var)


