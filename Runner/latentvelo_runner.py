from Runner.BaseRunner import BaseRunner
import models.latentvelo as ltv
import scvelo as scv
import torch as th
import matplotlib.pyplot as plt

class LatentVeloRunner(BaseRunner):
    def __init__(self, adata, batch_correction, device, zr_dim, h_dim=None, label=None, batch_key=None, latent_adata=None, pp_choice=0, n_hvg=2000, compute_umap=False, save_dir="logs/latentvelo", latent_dim=20, batch_size=100, epochs=50, grad_clip=100, annotated=False, t_prior=False, time_reg_weight=3, time_reg_decay=5, is_data_simu=False):
        self.zr_dim = zr_dim
        self.h_dim = h_dim if h_dim is not None else zr_dim
        self.batch_correction = batch_correction
        self.batch_key = batch_key
        self.compute_umap = compute_umap
        super().__init__(model_name="latentvelo", adata=adata, pp_choice=pp_choice, n_hvg=n_hvg, save_dir=save_dir, label=label, device=device)
        self.latent_adata = latent_adata # NOTE: neighbors need to be computed before embedding velocity in latent space
        self.v_latent_key = "spliced_velocity" # latent adata
        self.umap_latent_key = "latent_umap" # latent adata
        self.latent_emb_key = "latent" # adata
        self.latent_velo_key = "latent_velocity" # adata
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.annotated = annotated
        self.t_prior = t_prior
        self.time_reg_weight = time_reg_weight
        self.time_reg_decay = time_reg_decay
        self.is_data_simu = is_data_simu

    # def preprocess_real(self):
    #     scv.pp.filter_genes(self.adata, min_counts=20)
    #     self.adata = ltv.latentvelo.utils.standard_clean_recipe(self.adata, batch_key=self.batch_key, celltype_key=self.label, umap=self.compute_umap, n_top_genes=self.n_hvg)

    # def preprocess_simulation(self):
    #     self.adata = ltv.latentvelo.utils.standard_clean_recipe(self.adata, batch_key=self.batch_key, celltype_key=self.label, normalize_library=False, n_top_genes=None, log=False, umap=self.compute_umap)
    
    def preprocess(self):
        if self.pp_choice == 0:
            scv.pp.filter_genes(self.adata, min_shared_counts=20) # NOTE: corrected on 2025-04-23
            if self.annotated:
                self.adata = ltv.latentvelo.utils.anvi_clean_recipe(self.adata, batch_key=self.batch_key, celltype_key=self.label, umap=self.compute_umap, n_top_genes=self.n_hvg)
            else:
                self.adata = ltv.latentvelo.utils.standard_clean_recipe(self.adata, batch_key=self.batch_key, celltype_key=self.label, umap=self.compute_umap, n_top_genes=self.n_hvg)
        elif self.pp_choice == 3:
            if self.annotated:
                self.adata = ltv.latentvelo.utils.anvi_clean_recipe(self.adata, batch_key=self.batch_key, celltype_key=self.label, umap=self.compute_umap, n_top_genes=None)
            else:
                self.adata = ltv.latentvelo.utils.standard_clean_recipe(self.adata, batch_key=self.batch_key, celltype_key=self.label, umap=self.compute_umap, n_top_genes=None)
        else:
            if self.is_data_simu:
                self.adata.obs['spliced_size_factor'] = self.adata.layers['spliced'].sum(1)
                self.adata.obs['unspliced_size_factor'] = self.adata.layers['unspliced'].sum(1)
            else:
                self.adata.obs['spliced_size_factor'] = self.adata.obs['initial_size_spliced'].values
                self.adata.obs['unspliced_size_factor'] = self.adata.obs['initial_size_unspliced'].values
            if self.annotated:
                self.adata = ltv.latentvelo.utils.anvi_clean_recipe(self.adata, batch_key=self.batch_key, celltype_key=self.label, normalize_library=False, n_top_genes=None, log=False, umap=self.compute_umap, moments=self.pp_choice==1)
            else:
                self.adata = ltv.latentvelo.utils.standard_clean_recipe(self.adata, batch_key=self.batch_key, celltype_key=self.label, normalize_library=False, n_top_genes=None, log=False, umap=self.compute_umap, moments=self.pp_choice==1)
    
    def set_model(self):
        """hyperparameters
        dyngen: latent_dim=20, zr_dim=1
        bonemarrow: latent_dim=20, zr_dim=2, h_dim=2
        dentategyrus: latent_dim=20 , zr_dim=3, h_dim=4
        erythroid: latent_dim=20, zr_dim=1, h_dim=1
        organoid: latent_dim=20, zr_dim=2, h_dim=3
        cerebellar: latent_dim=20, zr_dim=1, h_dim=2
        """
        n_batches = len(self.adata.obs[self.batch_key].unique()) if self.batch_correction else 1
        if self.annotated:
            celltypes = len(self.adata.obs[self.label].unique())
            self.model = ltv.latentvelo.models.AnnotVAE(observed=self.adata.n_vars, celltypes=celltypes, batch_correction=self.batch_correction, batches=n_batches, zr_dim=self.zr_dim, h_dim=self.h_dim, latent_dim=self.latent_dim,
                                                        exp_time=self.t_prior, time_reg=self.t_prior, time_reg_weight=self.time_reg_weight, time_reg_decay = self.time_reg_decay)
        else:
            self.model = ltv.latentvelo.models.VAE(observed=self.adata.n_vars, batch_correction=self.batch_correction, batches=n_batches, zr_dim=self.zr_dim, h_dim=self.h_dim, latent_dim=self.latent_dim,
                                                   exp_time=self.t_prior, time_reg=self.t_prior, time_reg_weight=self.time_reg_weight, time_reg_decay = self.time_reg_decay)
    
    def train(self):
        """hyperparameters
        dyngen: batch_size = 100, epochs=50
        bonemarrow: batch_size = 100, epochs=50, grad_clip=100
        dentategyrus: batch_size = 100, epochs=50, grad_clip=100
        erythroid: batch_size = 100, epochs=25, grad_clip=100
        organoid: batch_size = 250, epochs=50, grad_clip=100
        cerebellar: batch_size = 1100, epochs=25
        """
        if self.annotated:
            self.records = ltv.train_anvi(self.model, self.adata, name=f"{self.save_dir}/params", batch_size=self.batch_size, grad_clip=self.grad_clip, epochs=self.epochs, random_seed=None)
        else:
            self.records = ltv.train(self.model, self.adata, name=f"{self.save_dir}/params", batch_size=self.batch_size, grad_clip=self.grad_clip, epochs=self.epochs, random_seed=None) # unset random seed on 2025-04-21

    def get_velocity(self):
        if self.latent_adata is None:
            self.get_results()
        return self.adata.layers["velo"]
    
    def get_latent_time(self):
        if self.latent_adata is None:
            self.get_results()
        return self.adata.obs["latent_time"]

    def save_adata(self):
        # plot records
        ltv.plot_history(*self.records)
        plt.savefig(f"{self.save_dir}/history.svg", format='svg')
        # save 
        self.latent_adata.write_h5ad(f"{self.save_dir}/latent_adata.h5ad")
        th.save(self.model.state_dict(), f"{self.save_dir}/model.pt")
        return super().save_adata()
    
    def get_results(self):
        self.latent_adata, self.adata = ltv.output_results(self.model, self.adata, gene_velocity=True) # NOTE: default embedding is umap
        # format adata
        self.adata.obsm[f'X_{self.latent_emb_key}'] = self.latent_adata.X
        self.adata.obsm[self.latent_velo_key] = self.latent_adata.layers["spliced_velocity"]
        # compute umap of latent 
         # NOTE: umap from original adata will be covered
        # umap = UMAP(n_neighbors=30, min_dist=1) # NOTE: default umap of scvelo can be used
        # latent_umap_key = f"{self.emb_latent_key}_umap"
        # self.latent_adata.obsm[latent_umap_key] = umap.fit_transform(self.latent_adata.X)
        # self.adata.obsm[latent_umap_key] = self.latent_adata.obsm[latent_umap_key]

    @property
    def adata2(self):
        # self.latent_adata.obsm[self.umap_ori_key] = self.latent_adata.obsm['X_umap'].copy()
        return self.latent_adata, self.v_latent_key
    
    @property
    def dimred2(self):
        if self.umap_latent_key not in self.latent_adata.obsm.keys():
            backup = False
            if 'X_umap' in self.latent_adata.obsm.keys():
                backup = True
                ori_umap = self.latent_adata.obsm['X_umap'].copy() # backup original umap
            scv.pp.neighbors(self.latent_adata, use_rep='X', n_neighbors=30)
            scv.tl.umap(self.latent_adata)
            self.latent_adata.obsm[f'X_{self.umap_latent_key}'] = self.latent_adata.obsm['X_umap'].copy()
            if backup:
                self.latent_adata.obsm['X_umap'] = ori_umap # restore original umap
        return self.umap_latent_key
        
    
    @property
    def emb_keys(self):
        return self.latent_emb_key, self.latent_velo_key

    # def add_annotations(self, bases, cluster_key, spatial_key=None, tkey_true=None):
    #     for basis in bases:
    #         self.latent_adata.obsm[f"X_{basis}"] = self.adata.obsm[f"X_{basis}"]
    #     self.latent_adata.obs[cluster_key] = self.adata.obs[cluster_key]
    #     if spatial_key is not None and f'X_{spatial_key}' not in self.latent_adata.obsm.keys():
    #         self.latent_adata.obsm[f'X_{spatial_key}'] = self.adata.obsm[f'X_{spatial_key}']
    #     if tkey_true is not None:
    #         self.latent_adata.obs[tkey_true] = self.adata.obs[tkey_true]
        # TODO: batch key