"""
    DP: Training on 2 GPUs
"""

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.benchmark
import torchvision
from torchvision.datasets import MNIST
from tqdm import tqdm
from train_single import CNN

assert torch.cuda.is_available(), 'CUDA is not available'
assert torch.cuda.device_count() >= 2, 'At least 2 GPUs are required'

mnist_path = '../data'

batchsize = 256
lr = 2e-3
epochs = 20
device = torch.device('cuda')

model = CNN().to(device)
model = nn.DataParallel(model)

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