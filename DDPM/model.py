import torch
from torch import nn
import math
import layers

class SinusoidalPositionEmbeddings( nn.Module ):
    def __init__( self, embedding_dim ):
        super().__init__()
        self.dim = embedding_dim

    def forward( self, time ):
        device = time.device

        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim-1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block( nn.Module ):
    def __init__( self, in_ch, out_ch, time_emb_dim, up=False ):
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward( self, x, t ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], 3, out_dim)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)

class UNet( nn.Module ):
    def __init__( self, channels_in=3, channels_out=3, time_dim=256 ):
        super().__init__()

        # 

        self.time_embedding = layers.SinusoidalPositionEmbeddings( time_dim )
        self.inc = layers.DoubleConv( channels_in, 64 )
        self.down1 = layers.Down(64, 128)
        self.sa1 = layers.SelfAttention(128,32)
        self.down2 = layers.Down(128,256)
        self.sa2 = layers.SelfAttention(256,16)
        self.down3 = layers.Down(256,256)
        self.sa3 = layers.SelfAttention(256,8)

        # Bottleneck

        self.bot1 = layers.DoubleConv(256,512)
        self.bot2 = layers.DoubleConv(512,512)
        self.bot3 = layers.DoubleConv(512,256)

        self.up1 = layers.Up(512,128)
        self.sa4 = layers.SelfAttention(128,16)
        self.up2 = layers.Up(256,64)
        self.sa5 = layers.SelfAttention(64,32)
        self.up3 = layers.Up(128,64)
        self.sa6 = layers.SelfAttention(64,64)
        self.outc = nn.Conv2d(64,channels_out,kernel_size=1)

    def forward( self, x, t ):
        t = self.time_embedding(t)
        x1 = self.inc(x)
        x2 = self.down1(x1,t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2,t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3,t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1( x4, x3, t )
        x = self.sa4(x)
        x = self.up2( x, x2, t )
        x = self.sa5(x)
        x = self.up3( x, x1, t )
        x = self.sa6(x)
        x = self.outc(x)

        return x

class Loss( nn.Module ):
    def __init__( self ):
        super().__init__()

        #self.loss = nn.L1Loss()
        self.loss = nn.MSELoss()

    def forward( self, noise, noise_pred ):
        return self.loss( noise, noise_pred )
