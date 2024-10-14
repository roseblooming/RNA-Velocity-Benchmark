import multiprocessing
from anndata import AnnData
import scvelo as scv
import time
import numpy as np
import scanpy as sc
from utils.monitor import monitor_memory_usage, monitor_gpu_memory_usage

class BaseRunner:
    def __init__(self, model_name, adata:AnnData, save_dir="logs/"):
        self.model_name = model_name
        self.adata = adata
        self.save_dir = save_dir
        
        # evaluate_results
        self.reconstruct_u = None
        self.reconstruct_s = None
        self.velocity = None
        self.latent_time = None

        self.run_time = None
        self.max_mem_usage = 0
        self.max_gpu_mem_usage = 0
        
        # kinetic parameters
        self.alpha = None
        self.beta = None
        self.gamma = None

        self.vkey = f"{self.model_name}_velocity"
        self.tkey = f"pred_t"
        self.u_hat_key = f"{self.model_name}_u_hat"
        self.s_hat_key = f"{self.model_name}_s_hat"
    
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
        if hasattr(self, 'device'):
            gpu_result_queue = multiprocessing.Queue()
        stop_event = multiprocessing.Event()
        cpu_monitor_process = multiprocessing.Process(target=monitor_memory_usage, args=(pid, cpu_result_queue, stop_event))
        if hasattr(self, 'device'):
            gpu_monitor_process = multiprocessing.Process(target=monitor_gpu_memory_usage, args=(pid, self.device, gpu_result_queue, stop_event))
        cpu_monitor_process.start()
        if hasattr(self, 'device'):
            gpu_monitor_process.start()
        
        t_start = time.time()
        self.train()
        t_end = time.time()
        self.run_time = t_end - t_start

        stop_event.set()
        cpu_monitor_process.join()
        if hasattr(self, 'device'):
            gpu_monitor_process.join()
        self.max_mem_usage = cpu_result_queue.get()
        if hasattr(self, 'device'):
            self.max_gpu_mem_usage = gpu_result_queue.get()
        else:
            self.max_gpu_mem_usage = 0
        
        # get params
        self.reconstruct_u, self.reconstruct_s = self.get_reconstruct_us()
        self.latent_time = self.get_latent_time()
        self.velocity = self.get_velocity()
        self.alpha, self.beta, self.gamma = self.get_fates()
        self.save_adata()
    
    def save_adata(self):
        self.adata.layers[self.vkey] = self.velocity
        self.adata.obs[self.tkey] = self.latent_time
        self._set_kinetic_param(f"{self.model_name}_alpha", self.alpha)
        self._set_kinetic_param(f"{self.model_name}_beta", self.beta)
        self._set_kinetic_param(f"{self.model_name}_gamma", self.gamma)
        if self.reconstruct_u is not None:
            self.adata.layers[self.u_hat_key] = self.reconstruct_u
        if self.reconstruct_s is not None:
            self.adata.layers[self.s_hat_key] = self.reconstruct_s
        import os
        os.makedirs(self.save_dir,exist_ok=True)
        sc.write(f"{self.save_dir}/{self.model_name}.h5ad", self.adata)

    # def draw_velocity(self, color=None, basis=None):
    #     vkey = f"{self.model_name}_velocity"
    #     scv.pp.neighbors(self.adata, basis=basis)
    #     scv.tl.velocity_graph(self.adata, vkey=vkey, basis=basis)
    #     scv.pl.velocity_embedding_stream(self.adata, vkey=vkey, basis=basis, color=color)
