import velovae as vv
import scvelo as scv
import scanpy as sc
import scipy.sparse

from Runner.BaseRunner import BaseRunner

#from BaseRunner import BaseRunner

class VeloVAE_Runner(BaseRunner):
    def __init__(self, adata, is_real, 
                 min_count_per_cell = 0,
                 min_genes_expressed = None,
                 compute_umap = True,
                 learning_rate = 2e-4,
                 learning_rate_ode = 5e-3,
                 learning_rate_post = 2e-4,
                 tmax = 20,
                 dim_z = 5,
                 device = 0, 
                 save_dir = "logs/velovae"):
        self.vae = None
        self.is_real = is_real
        self.min_count_per_cell = min_count_per_cell
        self.min_genes_expressed = min_genes_expressed
        self.compute_umap = compute_umap
        self.tmax = tmax
        self.dim_z = dim_z
        self.device = device
        self.infered=False
        self.config={
            'learning_rate': learning_rate,
            'learning_rate_ode': learning_rate_ode,
            'learning_rate_post': learning_rate_post
        }
        super().__init__(f"velovae", adata, save_dir)
    
    def preprocess_real(self):
        if 'neighbors' in self.adata.uns.keys():
            del self.adata.uns['neighbors']
        vv.preprocess(self.adata,
                      n_gene=self.adata.shape[1],
                      min_count_per_cell=self.min_count_per_cell,
                      min_genes_expressed=self.min_genes_expressed,
                      compute_umap=self.compute_umap
                      )
    
    def preprocess_simulation(self):
        scv.pp.normalize_per_cell(self.adata, enforce=True)
        scv.pp.log1p(self.adata)

        sc.pp.pca(self.adata, n_comps=30)
        sc.pp.neighbors(self.adata, n_neighbors=30)
        scv.pp.moments(self.adata, n_pcs=30, n_neighbors=30)

        scv.tl.umap(self.adata)

    def preprocess(self):
        if self.is_real:
            self.preprocess_real()
        else:
            self.preprocess_simulation()
        self.adata.layers["spliced"]=scipy.sparse.csr_matrix(self.adata.layers["spliced"])
        self.adata.layers["unspliced"]=scipy.sparse.csr_matrix(self.adata.layers["unspliced"])

    def set_model(self):
        self.vae=vv.VAE(self.adata,
                        self.tmax,
                        self.dim_z,
                        device=f'cuda:{self.device}'
                        )
        
    def get_reconstruct_us(self):
        self.u_hat_key = "vae_uhat"
        self.s_hat_key = "vae_shat"
        if not self.infered:
            self.vae.save_anndata(self.adata, "vae", file_path=self.save_dir)
            self.infered = True
        return self.adata.layers[self.u_hat_key], self.adata.layers[self.s_hat_key]

    def get_fates(self):
        if not self.infered:
            self.vae.save_anndata(self.adata, "vae", file_path=self.save_dir)
            self.infered = True
        alpha = self.adata.var[f"vae_alpha"]
        beta = self.adata.var[f"vae_beta"]
        gamma = self.adata.var[f"vae_gamma"]
        return alpha, beta, gamma
    
    def get_velocity(self):
        if not self.infered:
            self.vae.save_anndata(self.adata, "vae", file_path=self.save_dir)
            self.infered = True
        return self.adata.layers["vae_velocity"]

    def get_latent_time(self):
        if not self.infered:
            self.vae.save_anndata(self.adata, "vae", file_path=self.save_dir)
            self.infered = True
        return self.adata.obs["vae_time"].values
    
    def save_adata(self):
        if not self.infered:
            self.vae.save_anndata(self.adata, "vae", file_path=self.save_dir)
            self.infered = True
        self.vae.save_model(self.save_dir, 'encoder', 'decoder')
        
        return super().save_adata()

    def train(self):
        self.vae.train(self.adata,
                       figure_path=self.save_dir,
                       config=self.config
                       )

if __name__ == "__main__":
    adata = sc.read_h5ad("/HDD1/xueweizhi/spvelo_data/simulate/bidirect/exp_noise/0.5/20240605_222510/data_simu.h5ad")
    adata = adata[:2000]
    runner = VeloVAE_Runner(adata, False, device=4)
    runner.run()
    print(adata)