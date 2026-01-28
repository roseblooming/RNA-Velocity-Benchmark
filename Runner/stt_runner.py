from Runner.BaseRunner import BaseRunner
import models.stt as stt
import scanpy as sc
import scvelo as scv
import warnings

class STT_Runner(BaseRunner):
    def __init__(self, adata, pp_choice=0, n_hvg=2000, label=None, spatial_key=None, n_states=None, adata_aggr=None, save_dir="logs/stt"):
        self.n_states = n_states
        self.vkey_aggr = 'vj'
        self.adata_aggr = adata_aggr
        self.spatial_key = spatial_key
        if spatial_key is not None and spatial_key not in adata.obsm.keys():
            if f'X_{spatial_key}' in adata.obsm.keys():
                self.spatial_key = f'X_{spatial_key}'
            else:
                self.spatial_key = None
                warnings.warn(f"Spatial key '{spatial_key}' not found in adata.obsm keys. Defaulting to None.")
        super().__init__(model_name="stt", adata=adata, pp_choice=pp_choice, n_hvg=n_hvg, save_dir=save_dir, label=label)
        self.tkey = None # stt does not infer latent time

    # def preprocess_real(self):
    #     # sc.pp.filter_genes(self.adata, min_cells=5)
    #     # sc.pp.normalize_total(self.adata, target_sum=1e4)
    #     # sc.pp.log1p(self.adata)
    #     # sc.pp.highly_variable_genes(self.adata, n_top_genes=3000)
    #     scv.pp.filter_and_normalize(self.adata, min_shared_counts=20, n_top_genes=self.n_hvg) # TODO: how to preprocess
    #     # sc.pp.highly_variable_genes(self.adata, flavor='seurat_v3', n_top_genes=2000)
    #     # self.adata = self.adata[:, self.adata.var.highly_variable]
    #     scv.pp.moments(self.adata, n_pcs=30, n_neighbors=30)

    # def preprocess_simulation(self):
    #     scv.pp.moments(self.adata, n_pcs=30, n_neighbors=30)

    def preprocess(self):
        # if 'neighbors' in self.adata.uns.keys():
        #     del self.adata.uns['neighbors']
        if self.pp_choice == 0:
            scv.pp.filter_and_normalize(self.adata, min_shared_counts=20, n_top_genes=self.n_hvg)
        elif self.pp_choice == 3:
            scv.pp.filter_and_normalize(self.adata)
        if self.pp_choice in [0, 1, 3]:
            scv.pp.moments(self.adata, n_pcs=30, n_neighbors=30)
        
        if self.label is None:
            sc.tl.leiden(self.adata, resolution=0.2) # TODO: leiden resolution
            self.label = "leiden" # TODO: joint leiden
        # u = self.adata.layers['unspliced']
        # s = self.adata.layers['spliced']
        # if 'toarray' in dir(u):
        #     u = u.toarray()
        #     s = s.toarray()
        # X_all = np.concatenate([u, s], axis=1)
        # adata_aggr = sc.AnnData(X_all)
        # sc.tl.pca(adata_aggr)
        # sc.pp.neighbors(adata_aggr)
        # sc.tl.leiden(adata_aggr, resolution=0.15)
        # self.adata.obs['joint_leiden'] = adata_aggr.obs['leiden'].values
        # if self.label is None:
        #     self.n_states = self.adata.obs['joint_leiden'].nunique()
        # else:
        #     self.n_states = max(self.adata.obs[self.label].nunique(), self.adata.obs['joint_leiden'].nunique())
            # self.n_states = self.adata.obs[self.label].nunique()
        if self.n_states is None:
            self.n_states = self.adata.obs[self.label].nunique()
        if self.spatial_key is not None:
            sc.pp.neighbors(self.adata, use_rep=self.spatial_key, key_added='spatial') # TODO: whether to compute spatial neighbors
        self.adata.obs['attractor'] = self.adata.obs[self.label].values

    def get_velocity(self):
        return self.adata.layers["velo"]
    
    def train(self):
        self.adata_aggr = stt.tl.dynamical_iteration(self.adata, n_states=self.n_states, n_iter=15, 
                                                     weight_connectivities=0.5, n_components=21, n_neighbors=50, 
                                                     thresh_ms_gene=0.2, use_spatial=(self.spatial_key is not None), spa_weight=0.3)
        
    def get_latent_time(self):
        return None
    
    def get_reconstruct_us(self):
        return None, None
    
    def get_fates(self):
        return None, None, None
    
    def save_adata(self):
        # self.adata.layers[self.vkey] = self.velocity
        if 'louvain_colors' in self.adata.uns.keys():
            del self.adata.uns['louvain_colors']
        if 'scc_colors' in self.adata.uns.keys():
            del self.adata.uns['scc_colors']
        if 'scc_anno_colors' in self.adata.uns.keys():
            del self.adata.uns['scc_anno_colors']
        # self.adata.uns['kernel'].write(f"{self.save_dir}/kernel.h5ad")
        del self.adata.uns['kernel']
        # self.adata.uns['kernel_connectivities'].write(f"{self.save_dir}/kernel_connectivities.h5ad")
        del self.adata.uns['kernel_connectivities']
        if 'kernel_spatial' in self.adata.uns.keys():
            # self.adata.uns['kernel_spatial'].write(f"{self.save_dir}/kernel_spatial.h5ad")
            del self.adata.uns['kernel_spatial']
        self.adata.uns['r2_keep_train'] = self.adata.uns['r2_keep_train'].to_frame()
        self.adata.uns['r2_keep_test'] = self.adata.uns['r2_keep_test'].to_frame()
        # import os
        # os.makedirs(self.save_dir,exist_ok=True)
        # self.adata.write(f"{self.save_dir}/{self.model_name}.h5ad")
        super().save_adata()
        self.adata_aggr.write(f"{self.save_dir}/{self.model_name}_aggr.h5ad")

    @property
    def adata2(self):
        # if self.spatial_key is not None:
        #     self.adata_aggr.obsm[self.spatial_key] = self.adata.obsm[self.spatial_key]
        # if self.label is not None:
        #     self.adata_aggr.obs[self.label] = self.adata.obs[self.label].values
        return self.adata_aggr, self.vkey_aggr
    
    # def add_annotations(self, bases, cluster_key, spatial_key=None, tkey_true=None):
    #     for basis in bases:
    #         self.adata_aggr.obsm[f'X_{basis}'] = self.adata.obsm[f'X_{basis}']
    #     self.adata_aggr.obs[cluster_key] = self.adata.obs[cluster_key]
    #     if spatial_key is not None and f'X_{spatial_key}' not in self.adata_aggr.obsm.keys():
    #         self.adata_aggr.obsm[f'X_{spatial_key}'] = self.adata.obsm[f'X_{spatial_key}']
    #     if tkey_true is not None:
    #         self.adata_aggr.obs[tkey_true] = self.adata.obs[tkey_true]