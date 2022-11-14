import torch
import torch.nn.functional as F

class NoiseScheduler :
    def __init__( self, timesteps, start, end ):
        self.timesteps = int(timesteps)
        self.start = start
        self.end = end
        self.betas = torch.linspace( self.start,
                                     self.end,
                                     self.timesteps )

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod( self.alphas, axis=0 )
        self.alphas_cumprod_prev = F.pad( self.alphas_cumprod[:-1], (1,0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt( 1.0 / self.alphas )
        self.sqrt_alphas_cumprod = torch.sqrt( self.alphas_cumprod )
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * ( 1.0 - self.alphas_cumprod_prev ) / ( 1.0 - self.alphas_cumprod )

    def get_index_from_list( self, vals, t, x_shape ):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample( self, x_0, t, device="cpu" ):

        noise = torch.randn_like( x_0 )
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) \
                * noise.to(device), noise.to(device)

    @torch.no_grad()
    def sample_timestamp( self, x, t, noise_pred ):
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas * ( x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod )
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)

        if t == 0 :
            return model_mean
        else :
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt( posterior_variance_t ) * noise


