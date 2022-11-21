import torch

class DiffusionTransform :
    def __init__( self, diffusion_tools, device ):
        self.diffusion_tools = diffusion_tools
        self.device = device

    def __call__( self, X, Y ):

        batch_size = len(X)
        t = self.diffusion_tools.sample_timesteps( batch_size )

        X_noisy, noise = self.diffusion_tools.noise_images( X, t )
        t = t.unsqueeze(-1).type(torch.float)

        return X_noisy, noise, t, Y
