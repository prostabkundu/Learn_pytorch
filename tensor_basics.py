import torch
#torch.empty(size) : uninitialized
x = torch.empty(1) #scalar
print(x)
x = torch.empty(3) # vector 1d
print(x)
x = torch.empty(2,3) #matrix 2D
print(x)
x = torch.empty(2,2,3) # tensor
print(x)

#torch.rand(size):random number between 0 and 1
x = torch.rand(5,3)
print(x)
# torch.zeros(size):fill with zero
#torch.ones(size) : fill with one
x = torch.zeros(5,3)
print(x)
x = torch.ones(5,3)
print(x)
#check size
print(x.size())
#check type
print(x.dtype)
#specify types : default torch.float32
x = torch.zeros(5,3,dtype=torch.float16)
print(x)
print(x.dtype)

#construct from data
xu = [5.5,3]
x = torch.tensor(xu)
print(x)

# requires_grad argument
# This will tell pytorch that it will need to calculate the gradients for this tensor
# later in your optimization steps
# i.e. this is a variable in your model that you want to optimize

x = torch.tensor(xu,requires_grad=True)
print(x)
#operation
y = torch.rand(2,2)
x = torch.rand(2,2)
#element wise addition
z = x + y
print(z)
#substraction
z = x - y
print(z)
#multiplication
z = x * y
print(z)
#division
z = x/y
print(z)
# Slicing
x = torch.rand(5,3)
print(x)
print(x[:,0])
print(x[1,:])
print(x[1,1])
#get actual item
print(x[1,1].item())
# reshape with torch.view()
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8)
print(x.size(),y.size(),z.size())
#numpy
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))
a.add_(1)
print(a)
print(b)# addressing same memory location

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)
a +=1
print(a)
print(b)
#addressing same memory location
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x,device = device)
    x = x.to(device)
    z = x + y
    z.to("cpu")



