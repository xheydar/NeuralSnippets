import torch

class DiffusionTransformAtt :
    def __init__( self, diffusion_tools ):
        self.diffusion_tools = diffusion_tools

    def __call__( self, X, Y ):
        X = torch.cat( X, dim=0 )

        Y = [ i.unsqueeze(0) for i in Y ]
        Y = torch.cat( Y, dim=0 ).to( torch.float32 )

        batch_size = len(X)
        t = self.diffusion_tools.sample_timesteps( batch_size )

        X_noisy, noise = self.diffusion_tools.noise_images( X, t )
        t = t.unsqueeze(-1).type(torch.float)

        return X_noisy, noise, t, Y
