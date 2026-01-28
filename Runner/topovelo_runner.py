from Runner.BaseRunner import BaseRunner
import scvelo as scv
import models.topovelo as tpv
import scanpy as sc

class TopoVeloRunner(BaseRunner):
    def __init__(self, adata, spatial_key, spatial_graph_method, is_data_simu, pp_choice=0, n_hvg=2000,
                 min_count_per_cell=None,
                 min_genes_expressed=None,
                 compute_umap = False, # True
                 learning_rate = 2e-4,
                 learning_rate_ode = 5e-3,
                 learning_rate_post = 2e-4,
                 tmax = 20,
                 dim_z = 5,
                 device = 0,
                 save_dir="logs/scvelo", label=None, t_prior=False):
        self.is_data_simu = is_data_simu
        self.vae = None
        if spatial_key is None:
            raise ValueError("spatial_key is None.")
        elif spatial_key in adata.obsm.keys():
            self.spatial_key = spatial_key
        elif f'X_{spatial_key}' in adata.obsm.keys():
            self.spatial_key = f'X_{spatial_key}'
        else:
            raise KeyError(f"spatial_key {spatial_key} not found in adata.obsm")
        self.spatial_graph_method = spatial_graph_method
        self.min_count_per_cell = min_count_per_cell
        self.min_genes_expressed = min_genes_expressed
        self.compute_umap = compute_umap
        self.tmax = tmax
        self.dim_z = dim_z
        self.infered = False
        self.config={
            'learning_rate': learning_rate,
            'learning_rate_ode': learning_rate_ode,
            'learning_rate_post': learning_rate_post
        } if is_data_simu else {}
        super().__init__(model_name=f"topovelo", adata=adata, pp_choice=pp_choice, n_hvg=n_hvg, save_dir=save_dir, label=label, device=device)
        self.spatial_graph_key = "spatial_graph"
        self.random_seed = 42
        self.t_prior = t_prior
    
    # def preprocess_real(self):
    #     # tpv.preprocess(self.adata,
    #     #                n_gene=2000, # self.adata.shape[1]
    #     #                spatial_key=self.spatial_key,
    #     #                min_count_per_cell=self.min_count_per_cell,
    #     #                min_genes_expressed=self.min_genes_expressed,
    #     #                compute_umap=self.compute_umap,
    #     #                min_shared_counts=20,
    #     #             #    use_highly_variable=None
    #     #                )
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
        
        # build graph
        if self.spatial_graph_method == 'KNN':
            tpv.build_spatial_graph(self.adata,
                                    spatial_key=self.spatial_key,
                                    graph_key=self.spatial_graph_key,
                                    dist_cutoff='auto'
                                    )
        elif self.spatial_graph_method == 'Delaunay':
            tpv.build_spatial_graph(self.adata,
                                    spatial_key=self.spatial_key,
                                    graph_key=self.spatial_graph_key,
                                    method='Delaunay'
                                    )
        else:
            raise ValueError(f"spatial graph method {self.spatial_graph_method} not supported.")
    
    def set_model(self):
        # if self.is_real:
        self.vae = tpv.VAE(self.adata, 
                            tmax=self.tmax, 
                            dim_z=self.dim_z,
                            device=f'cuda:{self.device}',
                            graph_decoder=True,
                            attention=True,
                            hidden_size = tuple(x * round(self.adata.n_vars / 200) for x in (50, 25, 50)),
                            init_method='tprior' if self.t_prior else 'steady',
                            init_key='tprior' if self.t_prior else None,
                            tprior='tprior' if self.t_prior else None,
                            init_ton_zero=not self.t_prior,
                            random_state=self.random_seed
                            )

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
    
    def get_velocity(self):
        if not self.infered:
            self.vae.save_anndata(self.adata, "topo", file_path=self.save_dir)
            self.infered = True
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
                       cluster_key=self.label, # added on 2025-04-08
                       figure_path=self.save_dir,
                       random_state=self.random_seed
                       )
        
    def save_adata(self):
        if not self.infered:
            self.vae.save_anndata(self.adata, "topo", file_path=self.save_dir)
            self.infered = True
        self.vae.save_model(self.save_dir, "encoder", "decoder")
        return super().save_adata()
    
    def postprocess(self):
        self.adata = self.adata[(self.adata.layers[self.vkey] != 0).any(axis=1)].copy()
        del self.adata.uns['neighbors']
        del self.adata.obsp[self.spatial_graph_key]
        return super().postprocess()

if __name__ == "__main__":
    adata = sc.read_h5ad("/HDD1/xueweizhi/spvelo_data/simulate/bidirect/exp_noise/0.5/20240605_222510/data_simu.h5ad")
    adata = adata[:2000]
    runner = TopoVeloRunner(adata, "X_coord", False, device=4)
    runner.run()
    print(adata)