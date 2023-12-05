import torch.nn as nn 
import timm

class Net( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.backbone = timm.create_model('resnet34', pretrained=False, features_only=True )
        self.head = nn.Sequential (
                nn.Flatten(),
                nn.Linear(512,10)
        )

    def forward( self, x ):
        x = self.backbone(x)[-1]
        x = self.head(x)
        return x

class Loss( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

    def forward( self, outputs, labels ):
        return self.criterion( outputs, labels )
