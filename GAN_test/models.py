import torch
import torch.nn as nn

class UpSample( nn.Module ):
    def __init__( self, ch_in, ch_out, stride, padding ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d( ch_in, ch_out, 4, stride, padding, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(True),
            nn.Conv2d( ch_out, ch_out, 3, 1, 1, bias=False ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(True),
            nn.Conv2d( ch_out, ch_out, 3, 1, 1, bias=False ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(True)
        )

    def forward( self, X ):
        return self.layers(X)

class DownSample( nn.Module ):
    def __init__( self, ch_in, ch_out, stride, padding ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d( ch_in, ch_out, 4, stride, padding, bias=False ),
            nn.BatchNorm2d( ch_out ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d( ch_out, ch_out, 3,1,1, bias=False ),
            nn.BatchNorm2d( ch_out ),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d( ch_out, ch_out, 3,1,1, bias=False ),
            nn.BatchNorm2d( ch_out ),
            nn.LeakyReLU(0.2,inplace=True)
        )

    def forward( self, X ):
        return self.layers(X)

class Encoder(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=64):
        super().__init__()

        self.layers = nn.Sequential(
            DownSample( nc, ngf, 2, 1 ),
            DownSample( ngf, ngf*2, 2, 1 ),
            DownSample( ngf*2, ngf*4, 2,1 )
            #nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
            #nn.ReLU(inplace=True),
            
            #nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 2),
            #nn.ReLU(inplace=True),
            
            # state size. (ndf*2) x 16 x 16
            #nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 4),
            #nn.ReLU(inplace=True),
            # state size. (ndf*4) x 8 x 8
        )

        self.mean = nn.Conv2d( ngf * 4, nz, 4,2,1, bias=False )
        #self.logvar = nn.Conv2d( ngf * 4, nz, 4,2,1, bias=False )

    def forward(self,X):
        X = self.layers(X)

        mu = self.mean(X)

        return mu
        logvar = self.logvar(X)
        
        return mu, logvar

class Sampler( nn.Module ):
    def __init__( self ):
        super().__init__()

    def forward( self, mu, logvar ):
        std = torch.exp( 0.5 * logvar )
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

class LossVAE( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.l1 = nn.L1Loss( reduction='mean' )

    def forward( self, x, mu, log_var, d ):

        l1 = self.l1( x, d )
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return l1 + kld

class Generator(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=64):
        super().__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            UpSample( nz, ngf*8, 1, 0 ),
            UpSample( ngf*8, ngf*4, 2,1 ),
            UpSample( ngf*4, ngf*2, 2,1),
            UpSample( ngf*2, ngf, 2,1),
            nn.ConvTranspose2d( ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()

            # state size. (ngf*8) x 4 x 4
            #nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 4),
            #nn.ReLU(True),
            
            # state size. (ngf*4) x 8 x 8
            #nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 2),
            #nn.ReLU(True),
            
            # state size. (ngf*2) x 16 x 16
            #nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf),
            #nn.ReLU(True),
            #nn.ConvTranspose2d( ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
            #nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

#netG.load_state_dict(torch.load('weights/netG_epoch_99.pth'))
#print(netG)

class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super().__init__()

        self.main = nn.Sequential(
            DownSample( nc, ndf, 2, 1 ),
            DownSample( ndf, ndf*2, 2, 1 ),
            DownSample( ndf*2, ndf*4, 2,1 ),

            # input is (nc) x 64 x 64
            #nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            #nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            #nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

class Loss( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.loss = nn.BCELoss()

    def forward( self, output, label ):
        return self.loss( output, label )

