import torch
from torch import nn

class Encoder( nn.Module ):
    def __init__( self, layer_sizes, latent_size ):
        super().__init__()

        layers = []

        for l1, l2 in zip( layer_sizes, layer_sizes[1:] ):
            layers.append( nn.Linear( l1,l2 ) )
            layers.append( nn.ReLU() )

        self.layers = nn.Sequential( *layers )

        self.mean = nn.Linear( layer_sizes[-1], latent_size )
        self.log_var = nn.Linear( layer_sizes[-1], latent_size )

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

class Decoder( nn.Module ):
    def __init__( self, layer_sizes, latent_size ):
        super().__init__()

        layer_sizes = layer_sizes[::-1]

        layers = []
        layers.append( nn.Linear(latent_size, layer_sizes[0]) )
        layers.append( nn.ReLU() )

        for l1, l2 in zip( layer_sizes, layer_sizes[1:] ):
            layers.append(nn.Linear(l1,l2))
            layers.append(nn.ReLU())

        layers = layers[:-1]

        self.layers = nn.Sequential( *layers )

    def forward( self, X ):
        return self.layers(X)

class Net( nn.Module ):
    def __init__( self, layer_sizes, latent_size ):
        super().__init__()

        self.encoder = Encoder( layer_sizes, latent_size )
        self.sampler = Sampler()
        self.decoder = Decoder( layer_sizes, latent_size )

    def forward( self, X ):
        mu, log_var = self.encoder(X)
        s = self.sampler( mu, log_var )
        d = self.decoder( s )
        return mu, log_var, d

class Loss( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.l1 = nn.L1Loss( reduction='sum' )

    def forward( self, x, mu, log_var, d ):

        l1 = self.l1( x, d )
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return l1 + kld


