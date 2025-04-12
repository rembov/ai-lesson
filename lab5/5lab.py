"""
Классификатор погон ВС РФ по родам войск
Предположим, человек пытается
разберись, кто важнее...
...он - или какой-то другой военнослужащий.
В армии таких сомнений нет.
Просто посмотрите на погоны
и не стоит волноваться
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
train_dataset = torchvision.datasets.ImageFolder(root='./data/train',
                                              transform=data_transforms)
test_dataset = torchvision.datasets.ImageFolder(root='./data/test',
                                             transform=data_transforms)
class_names = train_dataset.classes
print("Классы:", class_names)
batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                         shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                        shuffle=False, num_workers=2)
net = torchvision.models.resnet18(pretrained=True)
for param in net.parameters():
    param.requires_grad = False
num_classes = 3
net.fc = nn.Linear(net.fc.in_features, num_classes)
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
num_epochs = 100
save_loss = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        save_loss.append(loss.item())
        
        if i % 10 == 9:
            print(f'Эпоха {epoch + 1}, шаг {i + 1}, loss: {running_loss / 10:.3f}')
            running_loss = 0.0
plt.figure(figsize=(10, 5))
plt.plot(save_loss)
plt.title('График функции потерь')
plt.xlabel('Итерации')
plt.ylabel('Loss')
plt.show()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Точность на тестовом наборе: {100 * correct / total:.2f}%')
torch.save(net.state_dict(), 'shoulder_straps_classifier.pt')
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
inputs, classes = next(iter(test_loader))
outputs = net(inputs.to(device))
_, preds = torch.max(outputs, 1)
plt.figure(figsize=(12, 8))
for i in range(min(8, inputs.size()[0])):
    plt.subplot(2, 4, i+1)
    imshow(inputs[i], f'Прогноз: {class_names[preds[i]]}\nРеальный: {class_names[classes[i]]}')
plt.tight_layout()
plt.show()