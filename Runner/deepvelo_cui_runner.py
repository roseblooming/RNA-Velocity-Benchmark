from Runner.BaseRunner import BaseRunner
from deepvelo import train, Constants
import scvelo as scv
from deepvelo.utils import velocity, update_dict, latent_time, save_model_and_config
from deepvelo.utils.preprocess import autoset_coeff_s


class DeepVeloCuiRunner(BaseRunner):
    def __init__(self, adata, pp_choice=0, n_hvg=2000,
                 device = 0, 
                 save_dir = "logs/deepvelo_cui", label=None):
        configs = {
            "name": "DeepVelo", # name of the experiment
            "loss": {"args": {"coeff_s": autoset_coeff_s(adata)}}, # Automatic setting of the spliced correlation objective
            "arch":{"args":{"pred_unspliced":True}},
            "n_gpu": device,
            "trainer": {"save_dir":save_dir}
        }
        self.configs = update_dict(Constants.default_configs, configs)
        super().__init__(model_name="deepvelo_cui", adata=adata, pp_choice=pp_choice, n_hvg=n_hvg, save_dir=save_dir, label=label, device=device)

    # def preprocess_real(self):
    #     scv.pp.filter_and_normalize(self.adata, min_shared_counts=20, n_top_genes=self.n_hvg)
    #     scv.pp.moments(self.adata, n_neighbors=30, n_pcs=30)

    # def preprocess_simulation(self):
    #     scv.pp.moments(self.adata, n_neighbors=30, n_pcs=30)

    def preprocess(self):
        if self.pp_choice == 0:
            scv.pp.filter_and_normalize(self.adata, min_shared_counts=20, n_top_genes=self.n_hvg)
        elif self.pp_choice == 3:
            scv.pp.filter_and_normalize(self.adata)
        if self.pp_choice in [0, 1, 3]:
            scv.pp.moments(self.adata, n_neighbors=30, n_pcs=30)
        velocity(self.adata)

    def train(self):
        self.trainer = train(self.adata, self.configs)

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
    
    def save_adata(self):
        save_model_and_config(self.trainer.model, self.configs, self.save_dir)
        return super().save_adata()