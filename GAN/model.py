import numpy as np
import torch
from torch import nn

class Upsample( nn.Module ):
    def __init__( self, dim, out_dim ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Upsample( scale_factor=2, mode='nearest' ),
            nn.Conv2d( dim, out_dim, 3, padding=1 )
        )

    def forward(self, X):
        return self.layers(X)

class Downsample( nn.Module ):
    def __init__( self, dim, out_dim ):
        super().__init__()

        self.layer = nn.Conv2d(dim, out_dim, 4, 2, 1)

    def forward( self, X ):
        return self.layer(X)

class Encoder( nn.Module ):
    def __init__( self, latent_size ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            nn.Conv2d(32,32,3,padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            Downsample(32,16),
            nn.Conv2d(16,16,3,padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            nn.Conv2d(16,16,3,padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            Downsample(16,8),
            nn.Flatten()
        )

        self.mean = nn.Linear( 392, latent_size )
        self.log_var = nn.Linear( 392, latent_size )

    def forward( self, X ):
        X = self.layers(X)

        mu = self.mean(X)
        log_var = self.log_var(X)

        return mu, log_var

class Sampler( nn.Module ):
    def __init__( self ):
        super().__init__()

    def forward( self, mu, log_var ):
        std = torch.exp( 0.5 * log_var )
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

class Generator( nn.Module ):
    def __init__( self, g_input_dim, g_output_shape ):
        super().__init__()

        c, h, w = g_output_shape

        self.base = nn.Sequential(
            nn.Linear( g_input_dim, 64 ), # 1,7,7
            nn.LeakyReLU(0.2),
            nn.Linear( 64, 3136 ),
            nn.LeakyReLU(0.2)
        )

        self.layers = nn.Sequential(
            nn.Conv2d( 64, 64, 3, padding=1 ),
            nn.LeakyReLU(0.2),
            nn.Conv2d( 64, 64, 3, padding=1 ),
            nn.LeakyReLU(0.2),
            Upsample(64, 32),
            nn.Conv2d( 32, 32, 3, padding=1 ),
            nn.LeakyReLU(0.2),
            nn.Conv2d( 32, 32, 3, padding=1 ),
            nn.LeakyReLU(0.2),
            Upsample(32, 16),
            nn.Conv2d(16,1, 3, padding=1 ),
        )

    def forward( self, X ):
        X = self.base(X)
        X = X.reshape(-1,64,7,7)
        X = self.layers(X)
        return X

class Discriminator( nn.Module ):
    def __init__( self, d_input_shape ):
        super().__init__()

        c, h, w = d_input_shape

        self.layers = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            nn.Conv2d(32,32,3,padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            Downsample(32,16),
            nn.Conv2d(16,16,3,padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            nn.Conv2d(16,16,3,padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            Downsample(16,8),
            nn.Flatten(),
            nn.Linear(392,1),
            nn.Sigmoid()
        )

    def forward( self, X ):
        return self.layers(X)

class PretrainLoss( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.l1 = nn.L1Loss( reduction='sum' )

    def forward( self, x, mu, log_var, d ):

        l1 = self.l1( x, d )
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return l1 + kld

class Loss( nn.Module ):
    def __init__( self ):
        super().__init__()
        self.bce_loss = nn.BCELoss()

    def forward( self, D_output, Y ):
        return self.bce_loss( D_output, Y )

