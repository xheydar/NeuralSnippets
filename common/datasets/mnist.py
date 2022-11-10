import torch
import torchvision

class mnist_dataset :
    def __init__( self, root, train, transform ):
        self.dataset = torchvision.datasets.MNIST(root='../data', train=train, transform=transform, download=True)
        self.dim = 784


