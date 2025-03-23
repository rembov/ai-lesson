import torch 
import random

x = torch.zeros(5,3,dtype=torch.int)
print(x)
x = torch.rand(5, 3)
print(x) 
x = x.to(dtype = torch.float32)
print(x)
z=x**3 
print(z)
sluch_chislo=random.randint(1,10)
print(sluch_chislo)
zz=z*sluch_chislo
print(zz)
x.requires_grad_(True)
eex = torch.exp(zz)
print(eex)
eex.backward(torch.ones_like(eex))
proiz=eex.grad
print(proiz)