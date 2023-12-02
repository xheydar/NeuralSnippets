import torch.nn as nn 

class Net( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.layers = nn.Sequential(
                    nn.Conv2d(3,6,5),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(6,16,5),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2),
                    nn.Flatten(1),
                    nn.Linear(16 * 5 * 5, 120),
                    nn.ReLU(),
                    nn.Linear(120,84),
                    nn.ReLU(),
                    nn.Linear(84,10)
                )

    def forward( self, x ):
        return self.layers(x)

class Loss( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

    def forward( self, outputs, labels ):
        return self.criterion( outputs, labels )
