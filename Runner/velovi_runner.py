import scvelo as scv
import velovi
from velovi import VELOVI
from Runner.BaseRunner import BaseRunner
# from BaseRunner import BaseRunner
import torch
import numpy as np
import scanpy as sc
# from anndata import AnnData
# from sklearn.preprocessing import MinMaxScaler
# from typing import Optional

class veloVI_Runner(BaseRunner):
    def __init__(self, adata, is_real, pp_choice=0, 
                 device = 0, 
                 save_dir = "logs/velovi"):
        self.vae=None
        # self.is_real = is_real
        self.device = device
        self.infered=False
        super().__init__(f"velovi", adata, is_real, pp_choice, save_dir)
    
    def preprocess_real(self):
        scv.pp.filter_and_normalize(self.adata, min_shared_counts=30, n_top_genes=2000) # FIXME: min_shared_counts
        scv.pp.moments(self.adata, n_pcs=30, n_neighbors=30)
        # self.adata = preprocess_data(self.adata)
    
    def preprocess_simulation(self):
        # scv.pp.neighbors(self.adata)
        scv.pp.moments(self.adata, n_pcs=30, n_neighbors=30)
        # scv.tl.velocity(self.adata, mode='deterministic', use_raw=True)
    
    def preprocess(self):
        if self.pp_choice == 0 and self.is_real:
            self.preprocess_real()
        elif self.pp_choice == 0 or self.pp_choice == 1:
            self.preprocess_simulation()
        self.adata = velovi.preprocess_data(self.adata)

    def set_model(self):
        # scv.tl.VELOVI.setup_anndata(self.adata, "Ms", "Mu")
        # self.vae=scv.tl.VELOVI(self.adata)
        VELOVI.setup_anndata(self.adata, "Ms", "Mu")
        self.vae=VELOVI(self.adata)

    def train(self):
        self.vae.train(accelerator="gpu",devices=[self.device])
    
    def get_fates(self):
        if not self.infered:
            add_velovi_outputs_to_adata(self.adata,self.vae)
            self.infered=True
        alpha=self.adata.var["fit_alpha"]
        beta=self.adata.var["fit_beta"]
        gamma=self.adata.var["fit_gamma"]
        return alpha, beta, gamma
    
    def get_velocity(self):
        if not self.infered:
            add_velovi_outputs_to_adata(self.adata,self.vae)
            self.infered=True
        return self.adata.layers["velocity"]
    
    def get_latent_time(self):
        if not self.infered:
            add_velovi_outputs_to_adata(self.adata,self.vae)
            self.infered=True
        cell_latent_time=self.adata.layers["latent_time_velovi"].mean(axis=1)
        return cell_latent_time
    
    def get_reconstruct_us(self):
        fit_s, fit_u = self.vae.get_expression_fit(self.adata)
        return fit_u, fit_s
    
    def save_adata(self):
        self.vae.save(self.save_dir, overwrite=True)
        return super().save_adata()
    
def add_velovi_outputs_to_adata(adata, vae):
    latent_time = vae.get_latent_time(n_samples=25)
    latent_time = latent_time.loc[adata.obs.index]
    velocities = vae.get_velocity(n_samples=25, velo_statistic="mean")

    t = latent_time
    scaling = 20 / t.max(0)

    adata.layers["velocity"] = velocities / scaling
    adata.layers["latent_time_velovi"] = latent_time

    adata.var["fit_alpha"] = vae.get_rates()["alpha"] / scaling
    adata.var["fit_beta"] = vae.get_rates()["beta"] / scaling
    adata.var["fit_gamma"] = vae.get_rates()["gamma"] / scaling
    adata.var["fit_t_"] = (
        torch.nn.functional.softplus(vae.module.switch_time_unconstr)
        .detach()
        .cpu()
        .numpy()
    ) * scaling
    adata.layers["fit_t"] = latent_time.values * scaling.to_numpy()[np.newaxis, :]
    adata.var['fit_scaling'] = 1.0
    
# def preprocess_data(
#     adata: AnnData,
#     spliced_layer: Optional[str] = "Ms",
#     unspliced_layer: Optional[str] = "Mu",
#     min_max_scale: bool = True,
#     filter_on_r2: bool = True,
# ) -> AnnData:
#     """Preprocess data.

#     This function removes poorly detected genes and minmax scales the data.

#     Parameters
#     ----------
#     adata
#         Annotated data matrix.
#     spliced_layer
#         Name of the spliced layer.
#     unspliced_layer
#         Name of the unspliced layer.
#     min_max_scale
#         Min-max scale spliced and unspliced
#     filter_on_r2
#         Filter out genes according to linear regression fit

#     Returns
#     -------
#     Preprocessed adata.
#     """
#     if min_max_scale:
#         scaler = MinMaxScaler()
#         adata.layers[spliced_layer] = scaler.fit_transform(adata.layers[spliced_layer])

#         scaler = MinMaxScaler()
#         adata.layers[unspliced_layer] = scaler.fit_transform(
#             adata.layers[unspliced_layer]
#         )

#     if filter_on_r2:
#         scv.tl.velocity(adata, mode="deterministic")

#         adata = adata[
#             :, np.logical_and(adata.var.velocity_r2 > 0, adata.var.velocity_gamma > 0)
#         ].copy()
#         adata = adata[:, adata.var.velocity_genes].copy()

#     return adata

if __name__ == "__main__":
    adata = sc.read_h5ad("/HDD1/xueweizhi/spvelo_data/simulate/bidirect/exp_noise/0.5/20240605_222510/data_simu.h5ad")
    adata = adata[:2000]
    runner = veloVI_Runner(adata, False, device=8)
    runner.run()
    print(adata)