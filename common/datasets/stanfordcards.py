import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

class stanfordcars_dataset :
    def __init__( self, root, image_size ):
        data_transforms = [
            transforms.Resize( (image_size, image_size) ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda( lambda t : (t*2)-1 )
        ]


        transform = transforms.Compose( data_transforms )

        dset_train = torchvision.datasets.StanfordCars( root=root, 
                                                        transform=transform, 
                                                        download=True )
        dset_test = torchvision.datasets.StanfordCars( root = root,
                                                       transform=transform,
                                                       download=True,
                                                       split='test' )

        self.dataset = torch.utils.data.ConcatDataset( [ dset_train, dset_test ] )

        self.dim = 3 * image_size * image_size
        self.shape = [3, image_size, image_size ]
        self.num_classes = 196

        self.reverse_transform = transforms.Compose([
            transforms.Lambda( lambda t : (t+1)/2 ),
            transforms.Lambda( lambda t : t.permute(1,2,0) ),
            transforms.Lambda( lambda t : t * 255 ),
            transforms.Lambda( lambda t : t.numpy().astype(np.uint8))
        ])


