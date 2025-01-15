from .autograd import Tensor
import torchvision

__all__ = ['load_mnist']

def load_mnist(download_path: str = './data'):
    """Load the MNIST dataset from torchvision.datasets.MNIST,
    and convert it to mytorch.Tensor format.

    Args:
        download_path (str, optional): The path to download and save the dataset. Defaults to './data'.

    Returns:
        dict: A dictionary containing the training and test sets with the following keys:
            - 'train': A dictionary with keys 'images' and 'labels' for the training images and labels
            - 'test': A dictionary with keys 'images' and 'labels' for the test images and labels
    """
    train_dataset = torchvision.datasets.MNIST(root=download_path, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root=download_path, train=False, download=True)

    return {'train': {'images': Tensor(train_dataset.data), 'labels': Tensor(train_dataset.targets)},
            'test': {'images': Tensor(test_dataset.data), 'labels': Tensor(test_dataset.targets)}}

def load_mnist_subset(n: int, download_path: str = './data'):
    train_dataset = torchvision.datasets.MNIST(root=download_path, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root=download_path, train=False, download=True)

    train_dataset.data = train_dataset.data[:n]
    train_dataset.targets = train_dataset.targets[:n]

    return {'train': {'images': Tensor(train_dataset.data), 'labels': Tensor(train_dataset.targets)},
            'test': {'images': Tensor(test_dataset.data), 'labels': Tensor(test_dataset.targets)}}
