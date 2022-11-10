import torch
from torch import nn

class Generator( nn.Module ):
    def __init__( self, g_input_dim, g_output_dim ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear( g_input_dim, 256 ),
            nn.LeakyReLU(0.2),
            nn.Linear( 256, 512 ),
            nn.LeakyReLU(0.2),
            nn.Linear( 512, 1024 ),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,g_output_dim),
            nn.Tanh()
        )

    def forward( self, X ):
        return self.layers(X)

class Discriminator( nn.Module ):
    def __init__( self, d_input_dim ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear( d_input_dim, 1024 ),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear( 1024, 512 ),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward( self, X ):
        return self.layers(X)

class Loss( nn.Module ):
    def __init__( self ):
        super().__init__()
        self.bce_loss = nn.BCELoss()

    def forward( self, D_output, Y ):
        return self.bce_loss( D_output, Y )

