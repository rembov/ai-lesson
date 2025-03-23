import pandas as pd # библиотека pandas нужна для работы с данными
import numpy as np # numpy для работы с векторами и матрицами
import torch
# Считываем данные 
# df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#     'machine-learning-databases/iris/iris.data', header=None)

df = pd.read_csv('data.csv')
# смотрим что в них
print(df.head())

# три столбца - это признаки, четвертый - целевая переменная (то, что мы хотим предсказывать)

# возьмем два признака, чтобы было удобне визуализировать задачу
X12 = df.iloc[:, [0,1,2]].values
prs=df.iloc[:, 4].values
# выделим целевую переменную в отдельную переменную
X = torch.tensor(X12, dtype=torch.float32)
y = torch.tensor([1 if label == "Iris-setosa" else -1 for label in prs ], dtype=torch.float32)
w = torch.rand(4, dtype=torch.float32, requires_grad=True)
speed_obuch = 0.01
epoxi = 100
def neuron(w, x):
    return 1 if (w[1] * x[0] + w[2] * x[1] + w[3] * x[2] + w[0]) >= 0 else -1
for epoch in range(epoxi):
    for i in range(len(X)):
        xi = X[i]
        target = y[i]
        predict = neuron(w, xi)
        with torch.no_grad():
            w_new = w.clone()
            w_new[1:] = w[1:] + speed_obuch * (target - predict) * xi
            w_new[0] = w[0] + speed_obuch * (target - predict)
            w.copy_(w_new)
cor = 0
for i in range(len(X)):
    if neuron(w, X[i]) == y[i]:
        cor += 1