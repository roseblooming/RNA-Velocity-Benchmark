from Runner.BaseRunner import BaseRunner
import scvelo as scv
import numpy as np

import models.topovelo as tpv
import scanpy as sc


class TopoVeloRunner(BaseRunner):
    def __init__(self, adata, spatial_key, is_real, 
                 min_count_per_cell=None,
                 min_genes_expressed=None,
                 compute_umap=True,
                 learning_rate = 2e-4,
                 learning_rate_ode = 5e-3,
                 learning_rate_post = 2e-4,
                 device = 0,
                 tmax = 20,
                 dim_z = 5,
                 save_dir="logs/scvelo"):
        self.vae = None
        self.is_real = is_real
        self.spatial_key = spatial_key
        self.min_count_per_cell = min_count_per_cell
        self.min_genes_expressed = min_genes_expressed
        self.compute_umap = compute_umap
        # self.learning_rate = learning_rate
        # self.learning_rate_ode = learning_rate_ode
        # self.learning_rate_post = learning_rate_post
        self.config={
            'learning_rate': learning_rate,
            'learning_rate_ode': learning_rate_ode,
            'learning_rate_post': learning_rate_post
        }
        self.device = device
        self.tmax = tmax
        self.dim_z = dim_z
        self.infered = False
        super().__init__(model_name=f"topovelo", adata=adata, save_dir=save_dir)
    
    def preprocess_real(self):
        if 'neighbors' in self.adata.uns.keys():
            del self.adata.uns['neighbors']
        tpv.preprocess(self.adata,
                       n_gene=self.adata.shape[1],
                       spatial_key=self.spatial_key,
                       min_count_per_cell=self.min_count_per_cell,
                       min_genes_expressed=self.min_genes_expressed,
                       compute_umap=self.compute_umap,
                       use_highly_variable=None
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
        # build graph
        tpv.build_spatial_graph(self.adata,
                                spatial_key=self.spatial_key,
                                graph_key='spatial_graph',
                                method='Delaunay')
        
    def set_model(self):
        if self.is_real:
            self.vae = tpv.VAE(self.adata, 
                               tmax=self.tmax, 
                               dim_z=self.dim_z, 
                               device=f'cuda:{self.device}',
                               graph_decoder=True,
                               attention=True)
                               # hidden_size=(50, 25, 50),
        else:
            self.vae = tpv.VAE(self.adata, 
                               tmax=20, 
                               dim_z=5, 
                               device=f"cuda:{self.device}",
                               graph_decoder=True,
                               attention=True,
                               reverse_gene_mode=False)

    def get_reconstruct_us(self):
        # self.u_hat_key = "topo_uhat"
        # self.s_hat_key = "topo_shat"
        if not self.infered:
            self.vae.save_anndata(self.adata, "topo", file_path=self.save_dir)
            self.infered = True
        return self.adata.layers["topo_uhat"], self.adata.layers["topo_shat"]

    def get_fates(self):
        if not self.infered:
            self.vae.save_anndata(self.adata, "topo", file_path=self.save_dir)
            self.infered = True
        alpha = self.adata.var[f"topo_alpha"]
        beta = self.adata.var[f"topo_beta"]
        gamma = self.adata.var[f"topo_gamma"]
        return alpha, beta, gamma
    
    def save_adata(self):
        if not self.infered:
            self.vae.save_anndata(self.adata, "topo", file_path=self.save_dir)
            self.infered = True
        self.vae.save_model(self.save_dir, "encoder", "decoder")
        return super().save_adata()
        
    def get_velocity(self):
        if not self.infered:
            self.vae.save_anndata(self.adata, "topo", file_path=self.save_dir)
            self.infered = True
        # self.vae.save_anndata(self.adata, "topo", file_path=self.save_dir)
        return self.adata.layers["topo_velocity"]

    def get_latent_time(self):
        if not self.infered:
            self.vae.save_anndata(self.adata, "topo", file_path=self.save_dir)
            self.infered = True
        return self.adata.obs["topo_time"].values

    def train(self):
        self.vae.train(self.adata, 
                       self.adata.obsp["spatial_graph"],
                       self.spatial_key,
                       config=self.config,
                       figure_path=self.save_dir
                       )


if __name__ == "__main__":
    adata = sc.read_h5ad("/HDD1/xueweizhi/spvelo_data/simulate/bidirect/exp_noise/0.5/20240605_222510/data_simu.h5ad")
    adata = adata[:2000]
    runner = TopoVeloRunner(adata, "X_coord", False, device=4)
    runner.run()
    print(adata)
