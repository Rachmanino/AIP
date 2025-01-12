"""
    Benchmark: training on single GPU.
"""

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.benchmark
import torchvision
from torchvision.datasets import MNIST
from tqdm import tqdm

assert torch.cuda.is_available(), 'CUDA is not available'

mnist_path = '../data'

batchsize = 256
lr = 2e-3
epochs = 20
device = 'cuda'

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten())
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10))

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
    
model = CNN().to(device)

train_dataset = MNIST(root=mnist_path, train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = MNIST(root=mnist_path, train=False, download=True, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
creterion = nn.CrossEntropyLoss()


if __name__ == '__main__':
    # train
    model.train()
    for epoch in tqdm(range(epochs)):
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = creterion(y_hat, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'Epoch {epoch}: loss = {sum(losses)/len(losses)}')


    # test
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            correct += (y_hat.argmax(1) == y).sum().item()
    acc = correct / len(test_dataset)
    print(f'Test acc = {acc:.4f}')

