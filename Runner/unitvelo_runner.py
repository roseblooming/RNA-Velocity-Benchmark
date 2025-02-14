import os
os.environ['TF_USE_LEGACY_KERAS'] = 'True'

from Runner.BaseRunner import BaseRunner
import unitvelo as utv
from unitvelo.utils import remove_dir
from unitvelo.velocity import Velocity
import numpy as np
import scanpy as sc
import pandas as pd
import scvelo as scv


class UniTVeloRunner(BaseRunner):
    def __init__(self, adata, is_real, pp_choice=0, 
                 device = 0, 
                 save_dir = "logs/unitvelo",
                 r2_adjust = False,
                 label = None):
        self.device = device
        self.model = None
        # self.is_real = is_real
        self.config = utv.config.Configuration()
        self.config.R2_ADJUST = r2_adjust
        self.config.GPU = device
        self.label = label
        super().__init__("unitvelo", adata, is_real, pp_choice, save_dir)

    def preprocess_real(self):
        scv.pp.filter_and_normalize(self.adata, 
                                    min_shared_counts=self.config.MIN_SHARED_COUNTS, 
                                    n_top_genes=self.config.N_TOP_GENES)
        scv.pp.moments(self.adata, 
                        n_pcs=self.config.N_PCS, 
                        n_neighbors=self.config.N_NEIGHBORS)
    
    def preprocess_simulation(self):
        scv.pp.moments(self.adata, 
                        n_pcs=self.config.N_PCS, 
                        n_neighbors=self.config.N_NEIGHBORS)
    
    def preprocess(self):
        if 'highly_variable' not in self.adata.var:
            self.adata.var['highly_variable'] = True
        if self.pp_choice == 0 and self.is_real:
            self.preprocess_real()
        elif self.pp_choice == 0 or self.pp_choice == 1:
            self.preprocess_simulation()
        # os.makedirs(self.save_dir, exist_ok=True)
        data_path = f'{self.save_dir}/dataset.h5ad'
        # self.adata.write(data_path, compression='gzip')
        # _, data_path = init_adata_and_logs(data_path, self.config)
        remove_dir(data_path, self.adata)
        self.adata.uns['datapath'] = data_path
        if self.label is None:
            sc.tl.leiden(self.adata, resolution=0.2) # TODO: leiden resolution
            self.adata.uns['label'] = 'leiden'
        else:
            self.adata.uns['label'] = self.label
        self.adata.uns['base_function'] = 'Gaussian'
        if self.config.BASIS is None:
            basis_keys = ["pca", "tsne", "umap"]
            basis = [key for key in basis_keys if f"X_{key}" in self.adata.obsm.keys()][-1]
        elif f"X_{self.config.BASIS}" in self.adata.obsm.keys():
            basis = self.config.BASIS
        else:
            raise ValueError('Invalid embedding parameter config.BASIS')
        self.adata.uns['basis'] = basis
    
    def set_model(self):
        self.model = Velocity(self.adata,config=self.config)
    
    def get_reconstruct_us(self):
        fit_u = pd.read_csv(f"{self.adata.uns['temp']}/fitu.csv").iloc[:, 1:]
        fit_s = pd.read_csv(f"{self.adata.uns['temp']}/fits.csv").iloc[:, 1:-1]
        return fit_u, fit_s
    
    def get_fates(self):
        alpha = np.ones(self.adata.shape[1]) * np.nan
        beta = self.adata.var['fit_beta']
        gamma = self.adata.var['fit_gamma']
        return alpha, beta, gamma
    
    def get_velocity(self):
        return self.adata.layers['velocity']
    
    def get_latent_time(self):
        return self.adata.obs['latent_time'].values
    
    def train(self):
        self.model.get_velo_genes()
        self.adata = self.model.fit_velo_genes(self.adata.uns['basis'], 0)
        # if 'examine_genes' in self.adata.uns.keys():
        #     from unitvelo.individual_gene import exam_genes
        #     exam_genes(self.adata, self.adata.uns['examine_genes'])
    
    def save_adata(self):
        #TODO save model
        return super().save_adata()
    
if __name__ == "__main__":
    adata = sc.read_h5ad("/HDD1/xueweizhi/spvelo_data/simulate/bidirect/exp_noise/0.5/20240605_222510/data_simu.h5ad")
    adata = adata[:2000]
    runner = UniTVeloRunner(adata, False)
    runner.run()
    print(adata)
