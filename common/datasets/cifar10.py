import torch
import torchvision

import torchvision
import torchvision.transforms as transforms


class cifar10_dataset :
    def __init__( self, root, image_size ):
        data_transforms = [
            transforms.Resize( (image_size, image_size) ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda( lambda t : (t*2)-1 )
        ]


        transform = transforms.Compose( data_transforms )

        dset_train = torchvision.datasets.cifar.CIFAR10( root=root, 
                                                         transform=transform, 
                                                         train=True,
                                                         download=True )
        dset_test = torchvision.datasets.cifar.CIFAR10( root = root,
                                                        transform=transform,
                                                        download=True,
                                                        train=False )

        self.dataset = torch.utils.data.ConcatDataset( [ dset_train, dset_test ] )

