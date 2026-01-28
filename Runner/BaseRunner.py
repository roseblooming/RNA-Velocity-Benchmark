import multiprocessing
from anndata import AnnData
import time
from numba import cuda
import scanpy as sc
from utils.monitor import monitor_cpu_memory_usage, monitor_gpu_memory_usage
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class BaseRunner:
    def __init__(self, model_name, adata:AnnData, pp_choice=0, n_hvg=2000, save_dir="logs/", label=None, device=-1):
        """constructor of BaseRunner

        Args:
            model_name (str): model name
            adata (AnnData): AnnData object
            is_real (bool): whether the data is real or simulated
            pp_choice (int, optional): preprocessing choice. 0 stands for standard preprocess, 1 stands for only moments, and 2 stands for no preprocess at all. Defaults to 0.
            n_hvg (int, optional): number of highly variable genes. Defaults to 2000.
            save_dir (str, optional): path to directory for saving results. Defaults to "logs/".
            label (str, optional): annotation of cell types. Defaults to None.
            device (int, optional): GPU card number for training. -1 stands for CPU only. Defaults to -1.
        """
        self.model_name = model_name
        self.adata = adata
        self.save_dir = save_dir
        # self.is_real = is_real
        self.pp_choice = pp_choice
        self.n_hvg = n_hvg
        self.label = label
        self.device = device
        
        # evaluate_results
        self.reconstruct_u = None
        self.reconstruct_s = None
        self.velocity = None
        self.latent_time = None

        self.cpu_time = None
        self.gpu_time = None
        self.max_cpu_mem_usage = None
        self.max_gpu_mem_usage = None
        
        # kinetic parameters
        self.alpha = None
        self.beta = None
        self.gamma = None

        self.vkey = f"{self.model_name}_velocity"
        self.tkey = f"pred_t"
        self.t_gene_key = None
        self.u_hat_key = f"{self.model_name}_u_hat"
        self.s_hat_key = f"{self.model_name}_s_hat"

        self.postprocessed = False # whether the adata has been postprocessed

        import os
        os.makedirs(self.save_dir, exist_ok=True)
    
    def preprocess(self):
        """
        preprocess adata
        """
        pass

    def train(self):
        """
        train model
        """
        pass

    def get_reconstruct_us(self):
        """
        return u, s. u and s are reconstructed by ode. shape: (n_obs, n_genes)
        """
        pass

    def get_velocity(self):
        """
        return velocity metrix (ds / dt). shape: (n_obs, n_genes).
        """
        pass

    def get_latent_time(self):
        """
        return latent time. shape: (n_obs) or (n_obs, 1)
        """
        pass

    def get_fates(self):
        """
        return (alpha, beta, gamma)
        """
        pass

    def _set_kinetic_param(self, key, value):
        """
        set kinetic parameters in adata.var / adata.varm according to shape of value
        """
        if isinstance(value, (int, float)) or len(value.shape) == 1:
            self.adata.var[key] = value
        else:
            self.adata.varm[key] = value

    def set_model(self):
        """
        define model
        """
        pass

    def run(self):
        # preprocess adata
        self.preprocess()
        # define model
        self.set_model()
        # train

        pid = multiprocessing.current_process().pid
        cpu_result_queue = multiprocessing.Queue()
        if self.device != -1:
            gpu_result_queue = multiprocessing.Queue()
        stop_event = multiprocessing.Event()
        cpu_monitor_process = multiprocessing.Process(target=monitor_cpu_memory_usage, args=(pid, cpu_result_queue, stop_event))
        if self.device != -1:
            gpu_monitor_process = multiprocessing.Process(target=monitor_gpu_memory_usage, args=(pid, self.device, gpu_result_queue, stop_event))
        cpu_monitor_process.start()
        if self.device != -1:
            gpu_monitor_process.start()
        
        try:
            # with cuda.gpus[0]:
            gpu_start = cuda.event(timing=True)
            gpu_end = cuda.event(timing=True)
            cuda.synchronize()
            gpu_start.record()
            t_start = time.process_time()
            self.train()
            t_end = time.process_time()
            gpu_end.record()
            cuda.synchronize()
            self.cpu_time = t_end - t_start
            self.gpu_time = cuda.event_elapsed_time(gpu_start, gpu_end) / 1000  # convert to seconds
        finally:
            stop_event.set()
            cpu_monitor_process.join()
            if self.device != -1:
                gpu_monitor_process.join()
        
        # get results
        self.max_cpu_mem_usage = cpu_result_queue.get()
        if self.device != -1:
            self.max_gpu_mem_usage = gpu_result_queue.get()
        else:
            self.max_gpu_mem_usage = 0
        
        # get params
        # NOTE: unused result extraction has been commented out
        # self.reconstruct_u, self.reconstruct_s = self.get_reconstruct_us()
        self.latent_time = self.get_latent_time()
        self.velocity = self.get_velocity()
        # self.alpha, self.beta, self.gamma = self.get_fates()
        self.save_adata()
    
    def save_adata(self):
        self.adata.layers[self.vkey] = self.velocity
        if self.latent_time is not None:
            self.adata.obs[self.tkey] = self.latent_time
        if self.alpha is not None:
            self._set_kinetic_param(f"{self.model_name}_alpha", self.alpha)
        if self.beta is not None:
            self._set_kinetic_param(f"{self.model_name}_beta", self.beta)
        if self.gamma is not None:
            self._set_kinetic_param(f"{self.model_name}_gamma", self.gamma)
        if self.reconstruct_u is not None:
            self.adata.layers[self.u_hat_key] = self.reconstruct_u
            self.adata.layers[self.s_hat_key] = self.reconstruct_s
        
        sc.write(f"{self.save_dir}/{self.model_name}.h5ad", self.adata)

    # def draw_velocity(self, color=None, basis=None):
    #     vkey = f"{self.model_name}_velocity"
    #     scv.pp.neighbors(self.adata, basis=basis)
    #     scv.tl.velocity_graph(self.adata, vkey=vkey, basis=basis)
    #     scv.pl.velocity_embedding_stream(self.adata, vkey=vkey, basis=basis, color=color)

    def postprocess(self):
        scaler = MinMaxScaler()
        if self.tkey is not None:
            self.adata.obs[self.tkey] = scaler.fit_transform(self.adata.obs[self.tkey].values.reshape(-1, 1))
        if self.t_gene_key is not None:
            mean_values = np.asarray(self.adata.layers[self.t_gene_key].mean(axis=1))
            self.adata.obs[self.t_gene_key] = scaler.fit_transform(mean_values.reshape(-1, 1))

    @property
    def processed_adata(self):
        """
        Get processed AnnData object.
        
        Returns:
            AnnData: Processed AnnData object.
        """
        if not self.postprocessed:
            self.postprocess()
            self.postprocessed = True
        return self.adata

    @property
    def adata2(self):
        """
        Get auxiliary AnnData object.
        
        Returns:
            tuple:
                - adata (AnnData): Auxiliary AnnData object.
                - vkey (str): Corresponding velocity key.
        """
        return None, None
    
    @property
    def dimred2(self):
        """
        Get dimensionality reduction representation for auxiliary adata, currently only for LatentVelo.
        - Caution: this will cover the original neighbor graph and umap of latent_adata.
        
        Returns:
            dimred_key (str): Dimensionality reduction key.
        """
        return None
    
    @property
    def emb_keys(self):
        """
        Get latent embedding key, currently only for LatentVelo.
        
        Returns:
            tuple:
                - latent_emb_key (str): Latent embedding key.
                - latent_velo_key (str): Latent velocity key.
        """
        return None, None
    
    # def cluster_cells_by_time(self, t_key, n_bins=5):
    #     """cluster cells by true time, for simulated data
    #     1. binning the time
    #     2. assign cluster label to each cell

    #     Args:
    #         t_key (str): key of true time in adata.obs
    #         n_bins (int, optional): number of bins. Defaults to 5.

    #     Returns:
    #         list: list of cluster edges
    #     """
    #     bins = np.linspace(self.adata.obs[t_key].min(), self.adata.obs[t_key].max(), n_bins + 1)
    #     bins[-1] = bins[-1] + 1
    #     labels = [str(i) for i in range(n_bins)]
    #     cluster_edges = [(str(i), str(i + 1)) for i in range(n_bins - 1)]
    #     self.adata.obs["t_cluster"] = pd.cut(self.adata.obs[t_key], bins=bins, labels=labels, right=False)
    #     return cluster_edges
    
    # def compute_spatial_graph(self, spatial_key, n_spatial_neighbors, spatial_graph_key='spatial_graph'):
    #     """compute spatial KNN graph

    #     Args:
    #         spatial_key (str): key of spatial coordinates in adata.obsm
    #         n_spatial_neighbors (int): number of neighbors
    #         spatial_graph_key (str, optional): key for spatial graph in adata.obsp. Defaults to 'spatial_graph'.
    #     """
    #     print('Computing spatial KNN graph.')
    #     X_pos = self.adata.obsm[spatial_key]
    #     nn = NearestNeighbors(n_neighbors=n_spatial_neighbors)
    #     nn.fit(X_pos)
    #     self.adata.obsp[spatial_graph_key] = nn.kneighbors_graph()
    #     self.adata.obsp['spatial_connectivities'] = nn.kneighbors_graph(mode='connectivity')
    #     self.adata.obsp['spatial_distances'] = nn.kneighbors_graph(mode='distance')

    # def add_annotations(self, bases, cluster_key, spatial_key=None, tkey_true=None):
    #     """add annotations of adata to auxiliary adata

    #     Args:
    #         bases (list[str]): list of basis keys in adata.obsm
    #         cluster_key (str): key of cluster labels in adata.obs
    #         spatial_key (str, optional): key of spatial coordinates in adata.obsm. Defaults to None.
    #         tkey_true (str, optional): key of true time in adata.obs. Defaults to None.
    #     """
    #     pass