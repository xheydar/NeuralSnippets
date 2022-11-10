import torch
import torchvision

class cifar10_dataset :
    def __init__( self, root, train, transform ):
        self.dataset = torchvision.datasets.cifar.CIFAR10(root='../data', train=train, 
                                                          transform=transform, download=True)
        self.dim = 3072


