import torch
import torchvision
import torchvision.transforms as transforms

class mnist_dataset :
    def __init__( self, root, train ):
        transform=transforms.Compose([
                           transforms.Resize(28),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,)),
                       ])

        self.dataset = torchvision.datasets.MNIST( root='../data', 
                                                   train=train, 
                                                   transform=transform, 
                                                   download=True )

        self.dim = 784
        self.shape = [1,28,28]


