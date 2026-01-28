from Runner.BaseRunner import BaseRunner
import models.velovgi as vgi
from pytorch_lightning import loggers

class VeloVGI_Runner(BaseRunner):
    def __init__(self, adata, batch_correction, sample, device, label=None, batch_key=None, pp_choice=0, n_hvg=2000, save_dir="logs/velovgi"):
        self.batch_correction = batch_correction
        self.batch_key = batch_key
        self.sample = sample
        super().__init__(model_name="velovgi", adata=adata, pp_choice=pp_choice, n_hvg=n_hvg, save_dir=save_dir, label=label, device=device)
        self.infered = False # state variable
        self.t_gene_key = "fit_t"
        self.tkey = None # velovgi does not infer cellular latent time

    # def preprocess_real(self):
    #     result = vgi.pp.preprocess(self.adata, batch_mode="batch" if self.batch_correction else None, batch_key=self.batch_key, sample_mode="random" if self.sample else None, n_top_genes=self.n_hvg)
    #     if self.sample:
    #         self.sampled_adata = result

    # def preprocess_simulation(self):
    #     result = vgi.pp.preprocess(self.adata, batch_mode="batch" if self.batch_correction else None, batch_key=self.batch_key, sample_mode="random" if self.sample else None, filter_and_normalize=False)
    #     if self.sample:
    #         self.sampled_adata = result
    
    def preprocess(self):
        result = vgi.pp.preprocess(self.adata, batch_mode="batch" if self.batch_correction else None, batch_key=self.batch_key, sample_mode="random" if self.sample else None, filter_and_normalize=self.pp_choice in [0, 3], min_shared_counts=20 if self.pp_choice==0 else None, n_top_genes=self.n_hvg if self.pp_choice==0 else None, moments=self.pp_choice in [0, 1, 3])
        if self.sample:
            self.sampled_adata = result

    def set_model(self):
        self.logger = loggers.TensorBoardLogger(save_dir=f"{self.save_dir}/log", name="base") # TensorBoard logging file
        vgi.tl.VELOVGI.setup_anndata(adata=self.sampled_adata if self.sample else self.adata, spliced_layer="Ms", unspliced_layer="Mu")
        self.model = vgi.tl.VELOVGI(adata=self.sampled_adata if self.sample else self.adata)

    def train(self):
        self.model.train(logger=self.logger, use_gpu=self.device)

    def get_results(self):
        vgi.tl.add_velovi_outputs_to_adata(adata=self.sampled_adata if self.sample else self.adata, vae=self.model)
        if self.sample:
            vgi.pp.moment_recover(self.adata, self.sampled_adata)
    
    def get_velocity(self):
        if not self.infered:
            self.get_results()
            self.infered = True
        return self.adata.layers["velocity"]
    
    def save_adata(self):
        if "sample_recover" in self.adata.uns.keys():
            import pickle
            # 需要单独保存uns中的sample_recover
            sample_recover_pkl_filename = f"{self.save_dir}/sample_recover.pkl"
            with open(sample_recover_pkl_filename, "wb") as f:
                pickle.dump(self.adata.uns["sample_recover"], f)
            del self.adata.uns["sample_recover"]
            print("save %s" % sample_recover_pkl_filename)
        # 布尔值转化为数字，方便保存
        is_sampled_key = "is_sampled"
        if is_sampled_key in self.adata.obs.columns:
            self.adata.obs[is_sampled_key] = self.adata.obs[is_sampled_key].apply(lambda x: 0 if x else 1)
        self.model.save(self.save_dir, overwrite=True)
        return super().save_adata()