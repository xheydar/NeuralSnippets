import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings( nn.Module ):
    def __init__( self, embedding_dim ):
        super().__init__()
        self.dim = embedding_dim

    def forward( self, t ):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.dim, 2, device=t.device).float() / self.dim)
        )

        pos_enc_a = torch.sin(t.repeat(1, self.dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, self.dim // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

class SelfAttention( nn.Module ):
    def __init__( self, channels, size ):
        super().__init__()

        self.channels = channels
        self.size = size

        self.mha = nn.MultiheadAttention( channels, 4, batch_first=True )
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear( channels, channels ),
            nn.GELU(),
            nn.Linear( channels, channels )
        )

    def forward( self, x ):
        return x

class DoubleConv( nn.Module ):
    def __init__( self, in_channels, out_chnnels, mid_channels=None, residual=False ):
        super().__init__()

        self.residual = residual
        if mid_channels is None :
            mid_channels = out_chnnels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d( in_channels, mid_channels, kernel_size=3, padding=1, bias=False ),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d( mid_channels, out_chnnels, kernel_size=3, padding=1, bias=False ),
            nn.GroupNorm(1, out_chnnels)
        )

    def forward( self, x ):
        if self.residual :
            return F.gelu( x + self.double_conv(x) )
        else :
            return self.double_conv(x)

class Down( nn.Module ):
    def __init__( self, in_channels, out_chnnels, emb_dim=256 ):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv( in_channels, in_channels, residual=True ),
            DoubleConv( in_channels, out_chnnels )
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear( emb_dim, out_chnnels )
        )

    def forward( self, x, t ):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
        return x + emb

class Up( nn.Module ):
    def __init__( self, in_channels, out_channels, emb_dim=256 ):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True )
        self.conv = nn.Sequential(
            DoubleConv( in_channels, in_channels, residual=True ),
            DoubleConv( in_channels, out_channels )
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward( self, x, skip_x, t ):
        x = self.up(x)
        x = torch.cat([ x, skip_x ], dim=1 )
        x = self.conv(x)

        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
