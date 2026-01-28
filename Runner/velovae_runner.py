from Runner.BaseRunner import BaseRunner
import scvelo as scv
import velovae as vv
import scanpy as sc

class VeloVAE_Runner(BaseRunner):
    def __init__(self, adata, is_data_simu, pp_choice=0, n_hvg=2000, 
                 min_count_per_cell = None, # 0
                 min_genes_expressed = None,
                 compute_umap = False, # True
                 learning_rate = 2e-4,
                 learning_rate_ode = 5e-3,
                 learning_rate_post = 2e-4,
                 tmax = 20,
                 dim_z = 5,
                 device = 0, 
                 save_dir = "logs/velovae", label = None, t_prior = False):
        self.is_data_simu = is_data_simu
        self.vae = None
        self.min_count_per_cell = min_count_per_cell
        self.min_genes_expressed = min_genes_expressed
        self.compute_umap = compute_umap
        self.tmax = tmax
        self.dim_z = dim_z
        self.infered=False
        self.config={
            'learning_rate': learning_rate,
            'learning_rate_ode': learning_rate_ode,
            'learning_rate_post': learning_rate_post
        } if is_data_simu else {}
        super().__init__(model_name=f"velovae", adata=adata, pp_choice=pp_choice, n_hvg=n_hvg, save_dir=save_dir, label=label, device=device)
        self.t_prior = t_prior
    
    # def preprocess_real(self):
    #     # vv.preprocess(self.adata,
    #     #               n_gene=2000, # self.adata.shape[1]
    #     #               min_count_per_cell=self.min_count_per_cell,
    #     #               min_genes_expressed=self.min_genes_expressed,
    #     #               min_shared_counts=20,
    #     #               compute_umap=self.compute_umap
    #     #               )
    #     scv.pp.filter_and_normalize(self.adata, min_shared_counts=20, n_top_genes=self.n_hvg)
    #     scv.pp.moments(self.adata, n_pcs=30, n_neighbors=30)
    #     if self.compute_umap:
    #         scv.tl.umap(self.adata)
    
    def preprocess_simulation(self):
        scv.pp.normalize_per_cell(self.adata, enforce=True)
        scv.pp.log1p(self.adata)

        # sc.pp.pca(self.adata, n_comps=30)
        # sc.pp.neighbors(self.adata, n_neighbors=30)
        scv.pp.moments(self.adata, n_pcs=30, n_neighbors=30)
        # if self.compute_umap:
        #     scv.tl.umap(self.adata)

    def preprocess(self):
        if self.is_data_simu:
            self.preprocess_simulation()
        else:
            if self.pp_choice == 0:
                scv.pp.filter_and_normalize(self.adata, min_shared_counts=20, n_top_genes=self.n_hvg)
            elif self.pp_choice == 3:
                scv.pp.filter_and_normalize(self.adata)
            if self.pp_choice in [0, 1, 3]:
                scv.pp.moments(self.adata, n_pcs=30, n_neighbors=30)
        if self.compute_umap:
            scv.tl.umap(self.adata)
        # TODO: check if this is necessary
        # self.adata.layers["spliced"]=scipy.sparse.csr_matrix(self.adata.layers["spliced"])
        # self.adata.layers["unspliced"]=scipy.sparse.csr_matrix(self.adata.layers["unspliced"])
    
    def set_model(self):
        self.vae=vv.VAE(self.adata,
                        self.tmax,
                        self.dim_z,
                        device=f'cuda:{self.device}',
                        hidden_size = tuple(x * round(self.adata.n_vars / 200) for x in (50, 25, 25, 50)),
                        init_method='tprior' if self.t_prior else 'steady',
                        init_key='tprior' if self.t_prior else None,
                        tprior='tprior' if self.t_prior else None,
                        init_ton_zero=not self.t_prior
                        )
    
    def get_reconstruct_us(self):
        # self.u_hat_key = "vae_uhat"
        # self.s_hat_key = "vae_shat"
        if not self.infered:
            self.vae.save_anndata(self.adata, "vae", file_path=self.save_dir)
            self.infered = True
        return self.adata.layers["vae_uhat"], self.adata.layers["vae_shat"]

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

    def train(self):
        self.vae.train(self.adata,
                       
                       config=self.config,
                       cluster_key=self.label, # added on 2025-04-08
                       figure_path=self.save_dir
                       )
        
    def save_adata(self):
        if not self.infered:
            self.vae.save_anndata(self.adata, "vae", file_path=self.save_dir)
            self.infered = True
        self.vae.save_model(self.save_dir, 'encoder', 'decoder') 
        return super().save_adata()

if __name__ == "__main__":
    adata = sc.read_h5ad("/HDD1/xueweizhi/spvelo_data/simulate/bidirect/exp_noise/0.5/20240605_222510/data_simu.h5ad")
    adata = adata[:2000]
    runner = VeloVAE_Runner(adata, False, device=4)
    runner.run()
    print(adata)