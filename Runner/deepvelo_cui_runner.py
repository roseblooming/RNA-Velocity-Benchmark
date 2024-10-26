from Runner.BaseRunner import BaseRunner
from deepvelo import train, Constants
import scvelo as scv
from deepvelo.utils import velocity, update_dict, latent_time
from deepvelo.utils.preprocess import autoset_coeff_s


class DeepVeloCuiRunner(BaseRunner):
    def __init__(self, adata, is_real, 
                 device = 0,
                 save_dir = "logs/deepvelo_cai"):
        self.is_real = is_real
        self.device = device
        configs = {
            "name": "DeepVelo", # name of the experiment
            "loss": {"args": {"coeff_s": autoset_coeff_s(adata)}}, # Automatic setting of the spliced correlation objective
            "arch":{"args":{"pred_unspliced":True}},
            "n_gpu": device,
            "trainer": {"save_dir":save_dir}
        }
        self.configs = update_dict(Constants.default_configs, configs)
        super().__init__(model_name="deepvelo_cai", adata=adata, save_dir=save_dir)

    def preprocess(self):
        if self.is_real:
            scv.pp.filter_and_normalize(self.adata, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(self.adata, n_neighbors=30, n_pcs=30)

    def train(self):
        velocity(self.adata)
        train(self.adata, self.configs)

    def get_velocity(self):
        return self.adata.layers['velocity']

    def get_latent_time(self):
        latent_time(self.adata)
        return self.adata.obs['latent_time'].values
    
    def get_fates(self):
        alpha = self.adata.layers['cell_specific_alpha'].mean(axis=0)
        beta = self.adata.layers['cell_specific_beta'].mean(axis=0)
        gamma = self.adata.layers['cell_specific_gamma'].mean(axis=0)
        return alpha, beta, gamma

    def get_reconstruct_us(self):
        fit_u = (self.adata.layers['cell_specific_alpha'] - self.adata.layers['velocity_unspliced']) / self.adata.layers['cell_specific_beta']
        fit_s = (self.adata.layers['cell_specific_alpha'] - self.adata.layers['velocity_unspliced'] - self.adata.layers['velocity']) / self.adata.layers['cell_specific_gamma']
        return fit_u, fit_s