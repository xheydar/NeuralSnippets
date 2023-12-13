import torch
import platform
import torchvision
import torchvision.transforms as transforms


class dataset :
    def __init__( self, cfg, transform, train=True ):

        root = cfg['root'][platform.system()]

        dataset = torchvision.datasets.CIFAR10( root=root, train=train, 
                                                download=True, transform=transform )

        self.dataset = dataset

    def __len__( self ):
        return len(self.dataset)

    def __getitem__( self, idx ):
        return self.dataset[idx]

    def get_loader( self, batch_size, shuffle, num_workers=2 ): 

        loader = torch.utils.data.DataLoader( self.dataset, batch_size=batch_size, 
                                              shuffle=shuffle, num_workers=num_workers )

        return loader
         
