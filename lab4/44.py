#я второй по списку,ррешал задачу на предсказание дохода
import pandas as pd
import torch
import torch.nn as nn
df = pd.read_csv('dataset_simple.csv')
X = torch.tensor(df[['age']].values, dtype=torch.float32)
y = torch.tensor(df[['income']].values, dtype=torch.float32)
class NNetRegression(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(NNetRegression, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
    
    def forward(self, X):
        return self.layers(X)

input_size = X.shape[1] 
hidden_size = 3  
output_size = 1  
net = NNetRegression(input_size, hidden_size, output_size)
lossFn = nn.L1Loss()  
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
epochs = 100
for i in range(epochs):
    pred = net(X)  
    loss = lossFn(pred.squeeze(), y)  
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  
    
    if i % 10 == 0:
        print(f'Ошибка на {i+1} итерации: {loss.item():.6f}')
with torch.no_grad():
    pred = net(X)

print('\nПредсказания:')
print(pred[:10])
err = torch.mean(abs(y - pred.T).squeeze())  
print('\nОшибка (MAE):')
print(err)

