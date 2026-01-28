from Runner.BaseRunner import BaseRunner
import scvelo as scv
import numpy as np


class scVeloRunner(BaseRunner):
    def __init__(self, model, adata, pp_choice=0, n_hvg=2000, save_dir="logs/scvelo", label=None):
        self.model = model
        super().__init__(model_name=f"scvelo_{model}", adata=adata, pp_choice=pp_choice, n_hvg=n_hvg, save_dir=save_dir, label=label)
        self.t_gene_key = 'fit_t'
    
    # def preprocess_real(self):
    #     scv.pp.filter_and_normalize(self.adata, min_shared_counts=20, n_top_genes=self.n_hvg)
    #     scv.pp.moments(self.adata, n_pcs=30, n_neighbors=30)
    
    # def preprocess_simulation(self):
    #     scv.pp.moments(self.adata, n_pcs=30, n_neighbors=30)
    
    def preprocess(self):
        if self.pp_choice == 0:
            scv.pp.filter_and_normalize(self.adata, min_shared_counts=20, n_top_genes=self.n_hvg)
        elif self.pp_choice == 3:
            scv.pp.filter_and_normalize(self.adata)
        if self.pp_choice in [0, 1, 3]:
            scv.pp.moments(self.adata, n_pcs=30, n_neighbors=30)

    def get_reconstruct_us(self):
        # TODO: compute according to ode.
        return None, None

    def get_fates(self):
        if self.model == "stochastic":
            alpha = np.ones(self.adata.shape[1]) * np.nan
            beta = np.ones(self.adata.shape[1]) * np.nan
            gamma = self.adata.var["velocity_gamma"]
        else:
            alpha = self.adata.var["fit_alpha"]
            beta = self.adata.var["fit_beta"]
            gamma = self.adata.var["fit_gamma"]
        return alpha, beta, gamma
        
    def get_velocity(self):
        return self.adata.layers["velocity"]

    def get_latent_time(self):
        return self.adata.obs["latent_time"].values

    def train(self):
        if self.model == "stochastic":
            scv.tl.velocity(self.adata, mode="stochastic")
            # scv.tl.latent_time(self.adata)
            scv.tl.velocity_pseudotime(self.adata)
        else:
            # scv.tl.recover_dynamics(self.adata, use_raw=True)
            scv.tl.recover_dynamics(self.adata,n_jobs=20)
            scv.tl.velocity(self.adata, mode="dynamical") # TODO: check diff_kinetics
            scv.tl.velocity_graph(self.adata,n_jobs=20)
            scv.tl.latent_time(self.adata)

    def postprocess(self):
        self.adata = self.adata[:, self.adata.var["velocity_genes"]].copy()
        return super().postprocess()

if __name__ == "__main__":
    adata = scv.datasets.simulation(n_obs=1000, n_vars=300)
    runner = scVeloRunner("stochastic", adata, False)
    runner.run()
    print(adata)
    adata = scv.datasets.simulation(n_obs=1000, n_vars=300)
    runner = scVeloRunner("dynamical", adata, False)
    runner.run()
    print(adata)
