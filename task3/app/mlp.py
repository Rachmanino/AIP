#TODO: 目前只搭出框架 需要完善


import mytorch
from mytorch import nn, Tensor
from mytorch.nn import functional as F
from mytorch.datasets import load_mnist, load_mnist_subset

if __name__ == '__main__':
    # Load the MNIST dataset
    dataset = load_mnist_subset(100)
    train_data, train_labels = dataset['train']['images'], dataset['train']['labels']
    test_data, test_labels = dataset['test']['images'], dataset['test']['labels']

    # Normalize the data
    train_data = train_data/ 255.
    test_data = test_data / 255.

    # Define the model
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            return x
        
    model = MLP()

    # Define the loss function and optimizer
    criterion = nn.CELoss()
    optimizer = nn.optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    for epoch in range(5):
        # Forward pass
        batch_data = train_data
        batch_labels = train_labels
        output = model(batch_data)
        loss = criterion(output, batch_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
