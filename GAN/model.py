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
    def __init__( self, nfeats, latent_size ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, nfeats, 1,1, bias=False ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nfeats, nfeats*2, 3,2,1, bias=False),
            nn.BatchNorm2d(nfeats*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nfeats*2, nfeats*4, 3,2,1, bias=False),
            nn.BatchNorm2d(nfeats*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nfeats*4, latent_size, 7,1, bias=False ),
        )

        self.mean = nn.Conv2d(latent_size, latent_size,1, bias=False )
        self.log_var = nn.Conv2d(latent_size, latent_size,1, bias=False )

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
    def __init__( self, nz, nfeats ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d( nz, nfeats*4, 7,1,0,bias=False ),
            nn.BatchNorm2d(nfeats*4),
            nn.ReLU(),
            nn.ConvTranspose2d( nfeats*4, nfeats*2, 4,2,1, bias=False ),
            nn.BatchNorm2d(nfeats*2),
            nn.ReLU(),
            nn.ConvTranspose2d( nfeats*2, nfeats, 4,2,1, bias=False ),
            nn.BatchNorm2d(nfeats),
            nn.Conv2d( nfeats, 1, 1, bias=False ),
        )


    def forward( self, X ):
        X = self.layers(X)
        return X

class Discriminator( nn.Module ):
    def __init__( self, nfeats ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, nfeats, 1,1, bias=False ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nfeats, nfeats*2, 3,2,1, bias=False),
            nn.BatchNorm2d(nfeats*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nfeats*2, nfeats*4, 3,2,1, bias=False),
            nn.BatchNorm2d(nfeats*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nfeats*4, 1, 7,1, bias=False ),
            nn.Flatten(),
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

