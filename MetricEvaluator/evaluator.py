from anndata import AnnData
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import scanpy as sc
import scvelo as scv
from prettytable import PrettyTable
from Runner.BaseRunner import BaseRunner
from Runner.scvelo_runner import scVeloRunner


########################################################
# Reference TopoVelo
########################################################
def _get_spatial_nbs(adata, spatial_graph_key, query_idx):
    # Get an adjacency list for cells in query_idx from a sparse adjacency matrix
    sub_graph = adata.obsp[spatial_graph_key][query_idx]
    return [np.where(sub_graph[i].A > 0)[1] for i in range(len(query_idx))]


def _cos_sim_sample(v_sample, v_neighbors, dt=None):
    res = cosine_similarity(v_neighbors, v_sample.reshape(1, -1)).flatten()
    if dt is not None:
        res = -np.abs(res) * (dt < 0) + res * (dt >= 0)
    return res


def _pearson_corr(v, v_neighbor):
    return np.corrcoef(v, v_neighbor)[0, 1:]


def keep_type(adata, nodes, target, k_cluster):
    """
    Select cells of targeted type

    Args:
        adata (anndata.AnnData):
            Anndata object.
        nodes (list):
            Indexes for cells
        target (str):
            Cluster name.
        k_cluster (str):
            Cluster key in adata.obs dataframe

    Returns:
        list:
             Selected cells.
    """

    return nodes[adata.obs[k_cluster][nodes].values == target]


# k-CBDir
def gen_cross_boundary_correctness(
    adata,
    k_cluster,
    k_velocity,
    cluster_edges,
    tkey,
    spatial_graph_key=None,
    k_hop=5,
    dir_test=False,
    x_emb="X_umap",
    gene_mask=None,
    n_prune=30,
    random_state=2022
):
    """Generalized Cross-Boundary Direction Correctness Score (A->B)

    Args:
        adata (:class:`anndata.AnnData`):
            Anndata object.
        k_cluster (str):
            Key to the cluster column in adata.obs.
        k_velocity (str):
            Key to the velocity matrix in adata.obsm.
        cluster_edges (list[tuple[str]]):
            Pairs of clusters has transition direction A->B
        tkey (str):
            Key to the cell time in adata.obs
        k_hop (int, optional):
            Number of steps to consider.
            CBDir will be computed for 1 to k-step neighbors.
            Defaults to 5.
        dir_test (bool, optional):
            Whether to subtract CBDir of random walk from CBDir of desired direction.
            Defaults to False.
        x_emb (str, optional):
            Low dimensional embedding in adata.obsm
            or original count matrix in adata.layers.
            Defaults to "X_umap".
        gene_mask (:class:`numpy.ndarray`, optional):
            Boolean array to filter out non-velocity genes. Defaults to None.
        n_prune (int, optional):
            Maximum number of neighbors to keep.
            This is necessary because number of neighbors grows exponentially.
            Defaults to 30.
        random_state (int, optional):
            Seed for random walk sampling. Defaults to 2022.

    Returns:
        tuple:

            - dict: all_scores indexed by cluster_edges or mean scores indexed by cluster_edges

            - :class:`numpy.ndarray`: Average score over all cells for all step numbers
    """
    # Use k-hop neighbors
    scores = {}
    x_emb_name = x_emb
    if x_emb in adata.obsm:
        x_emb = adata.obsm[x_emb]
        if x_emb_name == "X_umap":
            v_emb = adata.obsm[f'{k_velocity}_umap']
        else:
            v_emb = adata.obsm[f'{k_velocity}_{x_emb_name[2:]}']
    else:
        x_emb = adata.layers[x_emb]
        v_emb = adata.layers[k_velocity]
        if gene_mask is None:
            gene_mask = ~np.isnan(v_emb[0])
        x_emb = x_emb[:, gene_mask]
        v_emb = v_emb[:, gene_mask]
    t = adata.obs[tkey].to_numpy()
    cell_labels = adata.obs[k_cluster].to_numpy()
    
    np.random.seed(random_state)
    for u, v in cluster_edges:
        sel = cell_labels == u
        if spatial_graph_key is None:
            nbs = adata.uns['neighbors']['indices'][sel]  # [n * 30]
        else:
            nbs = _get_spatial_nbs(adata, spatial_graph_key, np.where(sel)[0])

        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, k_cluster), nbs)
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]
        t_points = t[sel]

        type_score = [[] for i in range(k_hop)]
        type_score_null = [[] for i in range(k_hop)]
        for x_pos, x_vel, nodes, t_i, all_nodes in zip(x_points, x_velocities, boundary_nodes, t_points, nbs):
            if len(nodes) == 0:
                continue
            position_dif = x_emb[nodes] - x_pos
            dt = t[nodes] - t_i
            dir_scores = _cos_sim_sample(x_vel, position_dif, dt)

            nodes_null = all_nodes if len(all_nodes) < n_prune else np.random.choice(all_nodes, n_prune)
            position_dif_null = x_emb[nodes_null] - x_pos
            dt_null = t[nodes_null] - t_i
            dir_scores_null = _cos_sim_sample(x_vel, position_dif_null, dt_null)

            # save 1-hop results
            type_score[0].append(np.nanmean(dir_scores))
            type_score_null[0].append(np.nanmean(dir_scores_null))
            # deal with k-hop neighbors when k > 1
            for k in range(k_hop-1):
                nodes = adata.uns['neighbors']['indices'][nodes].flatten()  # [num (k-1)-hop neighbors * 30]
                nodes = keep_type(adata, nodes, v, k_cluster)
                nodes = np.unique(nodes)

                position_dif = x_emb[nodes] - x_pos
                dt = t[nodes] - t_i
                dir_scores = _cos_sim_sample(x_vel, position_dif, dt)

                if len(nodes) > n_prune:
                    idx_sort = np.argsort(dir_scores)
                    nodes = nodes[idx_sort[-n_prune:]]
                    dir_scores = dir_scores[idx_sort[-n_prune:]]
                type_score[k+1].append(np.nanmean(dir_scores))
            # Compute the same k-hop metric for neigbhors not in the descent v
            if dir_test and len(nodes_null) > 0:
                for k in range(k_hop-1):
                    # [num (k-1)-hop neighbors * 30]
                    nodes_null = adata.uns['neighbors']['indices'][nodes_null].flatten()
                    nodes_null = np.unique(nodes_null)
                    nodes_null = nodes_null if len(all_nodes) < n_prune else np.random.choice(nodes_null, n_prune)
                    position_dif_null = x_emb[nodes_null] - x_pos
                    dt_null = t[nodes_null] - t_i
                    dir_scores_null = _cos_sim_sample(x_vel, position_dif_null, dt_null)
                    if len(nodes_null) > n_prune:
                        idx_sort = np.argsort(dir_scores_null)
                        nodes_null = nodes_null[idx_sort[-n_prune:]]
                        dir_scores_null = dir_scores_null[idx_sort[-n_prune:]]
                    type_score_null[k+1].append(np.nanmean(dir_scores_null))
        mean_type_score = np.array([np.nanmean(type_score[i]) for i in range(k_hop)])
        mean_type_score_null = np.array([np.nanmean(type_score_null[i]) for i in range(k_hop)])
        mean_type_score_null[np.isnan(mean_type_score_null)] = 0.0
        scores[f'{u} -> {v}'] = (mean_type_score - mean_type_score_null if dir_test else mean_type_score)

    return scores, np.mean(np.stack([sc for sc in scores.values()]), 0)


def spatial_velocity_consistency(adata, vkey, spatial_graph_key, gene_mask=None):
    """Velocity Consistency as reported in scVelo paper

    Args:
        adata (:class:`anndata.AnnData`):
            Anndata object.
        vkey (str):
            key to the velocity matrix in adata.obsm.
        spatial_graph_key (:class:`scipy.sparse_matrix`):
            Key in .obsp for the spatial graph
        gene_mask (:class:`numpy.ndarray`, optional):
            Boolean array to filter out genes. Defaults to None.

    Returns:
        float: Average score over all cells.
    """
    graph = adata.obsp[spatial_graph_key]
    nbs = [np.where(graph[i].A > 0)[1] for i in range(graph.shape[0])]

    velocities = adata.layers[vkey]
    nan_mask = ~np.isnan(velocities[0]) if gene_mask is None else gene_mask
    velocities = velocities[:, nan_mask]

    consistency_score = []
    for ith, nbs_i in enumerate(nbs):
        if len(nbs_i) < 1:
            continue;
        consistency_score.append(_pearson_corr(velocities[ith], velocities[nbs_i]).mean())
    # adata.obs[f'{vkey}_consistency'] = consistency_score
    return np.nanmean(consistency_score)


def global_moran_I(vals, nbs):
    # nbs: adjacency list
    mean = vals.mean()
    N = len(vals)
    W = np.sum([len(x) for x in nbs])
    num = np.sum([np.sum((vals[i]-mean)*(vals[nbs[i]]-mean)) for i in range(len(nbs))])
    denom = np.sum((vals-mean)**2)
    return (N / W)*(num / denom)


def spatial_time_consistency(adata,
                             tkey,
                             spatial_graph_key):
    """Computes local Moran's I as a measure of spatial consistency.

    Args:
        adata (:class:`anndata.AnnData`):
            Anndata object.
        tkey (str):
            Key for latent time.
        spatial_graph_key (spatial_graph_key):
            Key for spatial graph.
    """
    graph = adata.obsp[spatial_graph_key]
    nbs = [np.where(graph[i].A > 0)[1] for i in range(graph.shape[0])]

    t = adata.obs[tkey].to_numpy()
    return global_moran_I(t, nbs)


def compute_spatial_graph(adata, spatial_key, n_spatial_neighbors):
    print('Computing spatial KNN graph.')
    X_pos = adata.obsm[spatial_key]
    nn = NearestNeighbors(n_neighbors=n_spatial_neighbors)
    nn.fit(X_pos)
    adata.obsp['spatial_graph'] = nn.kneighbors_graph()
    adata.obsp['connectivities'] = nn.kneighbors_graph(mode='connectivity')
    adata.obsp['distances'] = nn.kneighbors_graph(mode='distance')


########################################################
# End Reference
########################################################


def velocity_accuracy(adata, vkey_pred="velocity_pred", vkey_true="velocity"):
    """
    we computed the average cosine similarity between inferred spatial velocity and the true cell velocity,
    which we refer to as “velocity accuracy”
    Args:
        adata (:class:`anndata.AnnData`):
            Anndata object.
        vkey_pred (str):
            Key for predicted velocity.
        vkey_true (str):
            Key for true velocity.
    """
    if vkey_pred in adata.layers:
        cosine_sim = cosine_similarity(adata.layers[vkey_pred], adata.layers[vkey_true])
        cosine_sim = cosine_sim.diagonal().mean()
    else:
        cosine_sim = cosine_similarity(adata.obsm[vkey_pred], adata.obsm[vkey_true])
        cosine_sim = cosine_sim.diagonal().mean()
    return cosine_sim


def cluster_cells_by_time(adata, t_key, n_bins=5):
    bins = np.linspace(0, adata.obs[t_key].max(), n_bins + 1)
    bins[-1] = bins[-1] + 1
    labels = [str(i) for i in range(n_bins)]
    cluster_edges = [(str(i), str(i + 1)) for i in range(n_bins - 1)]
    adata.obs["t_cluster"] = pd.cut(adata.obs[t_key], bins=bins, labels=labels, right=False)
    return cluster_edges


def abundance_mse_us(adata: AnnData, ukey_pred, ukey_true, skey_pred, skey_true):
    u_mse = mean_squared_error(adata.layers[ukey_true], adata.layers[ukey_pred])
    s_mse = mean_squared_error(adata.layers[skey_true], adata.layers[skey_pred])
    return u_mse, s_mse


class VelocityMetric:
    def __init__(self, 
                 cbdir=None, 
                 time_corr_spearman=None, 
                 time_corr_spearman_cell_gene=None,
                 velocity_acc=None, 
                 spatial_time_consist=None, 
                 spatial_velo_consist=None,
                 train_time=None,
                 title=None,
                 abundance_mse=None
                 ):
        self.cbdir_edge = None
        self.cbdir_mean = None
        self.cbdir = cbdir
        self.time_corr_spearman = time_corr_spearman
        self.time_corr_spearman_cell_gene = time_corr_spearman_cell_gene
        self.velocity_acc = velocity_acc
        self.spatial_time_consist = spatial_time_consist
        self.spatial_velo_consist = spatial_velo_consist
        self.train_time = train_time
        self.title = title if title is not None else "Metric"
        self.abundance_mse_u = None
        self.abundance_mse_s = None
        self.abundance_mse = abundance_mse

        self.velo_na_ratio = None
        self.velo_zero_ratio = None

        self.mem_usage = None
        self.gpu_mem_usage = None
    
    @property
    def cbdir(self):
        return self.cbdir_edge, self.cbdir_mean

    @cbdir.setter
    def cbdir(self, value):
        if value is None:
            self.cbdir_edge = None
            self.cbdir_mean = None
        else:
            self.cbdir_edge, self.cbdir_mean = value

    @property
    def abundance_mse(self):
        return self.abundance_mse_u, self.abundance_mse_s

    @abundance_mse.setter
    def abundance_mse(self, value):
        if value is None:
            self.abundance_mse_u = None
            self.abundance_mse_s = None
        else:
            self.abundance_mse_u, self.abundance_mse_s = value

    def __str__(self):
        table = PrettyTable(["metric", "value"], title=self.title)
        table.add_row(["cbdir_mean", str([round(i, 4) for i in self.cbdir_mean]) if self.cbdir_mean is not None else None])
        table.add_row(["time_corr_spearman", self.time_corr_spearman])
        table.add_row(["time_corr_spearman_cell_gene", self.time_corr_spearman_cell_gene])
        table.add_row(["velocity_acc", round(self.velocity_acc, 4) if self.velocity_acc is not None else None])
        table.add_row(["spatial_time_consist", round(self.spatial_time_consist, 4) if self.spatial_time_consist is not None else None])
        table.add_row(["spatial_velo_consist", round(self.spatial_velo_consist, 4) if self.spatial_velo_consist is not None else None])
        table.add_row(["train_time", round(self.train_time, 2) if self.train_time is not None else None])
        table.add_row(["velo_na_ratio", round(self.velo_na_ratio, 4) if self.velo_na_ratio is not None else None])
        table.add_row(["velo_zero_ratio", round(self.velo_zero_ratio, 4) if self.velo_zero_ratio is not None else None])
        table.add_row(["abundance_mse_u", round(self.abundance_mse_u, 4) if self.abundance_mse_u is not None else None])
        table.add_row(["abundance_mse_s", round(self.abundance_mse_s, 4) if self.abundance_mse_s is not None else None])
        table.add_row(["mem_usage", round(self.mem_usage, 2) if self.mem_usage is not None else None])
        table.add_row(["gpu_mem_usage", round(self.gpu_mem_usage) if self.gpu_mem_usage is not None else None])
        return table.__str__()

    def to_series(self):
        metric_dict = {"cbdir_mean": self.cbdir_mean,
                       "cbdir_edge": self.cbdir_edge,
                       "time_corr_spearman": self.time_corr_spearman,
                       "time_corr_spearman_cell_gene": self.time_corr_spearman_cell_gene,
                       "velocity_acc": self.velocity_acc,
                       "train_time": self.train_time,
                       "spatial_time_consist": self.spatial_time_consist,
                       "spatial_velo_consist": self.spatial_velo_consist,
                       "velo_na_ratio": self.velo_na_ratio,
                       "velo_zero_ratio": self.velo_zero_ratio,
                       "abundance_mse_u": self.abundance_mse_u,
                       "abundance_mse_s": self.abundance_mse_s,
                       "mem_usage": self.mem_usage,
                       "gpu_mem_usage": self.gpu_mem_usage
                       }
        return pd.Series(metric_dict)


def evaluate_metrics(adata, cluster_edges=None, cluster_key="t_cluster", 
                     t_key_pred="pred_t", 
                     t_key_true="true_t",
                     v_key_pred="velocity_pred", 
                     v_key_true="true_velocity", 
                     spatial_graph_key="spatial_graph", 
                     spatial_key="X_coord",
                     n_neighbors=30,
                     title=None):
    metric = VelocityMetric(title=title)
    metric.time_corr_spearman = adata.obs[[t_key_pred, t_key_true]].corr(method="spearman").loc[t_key_true, t_key_pred]
    if spatial_key not in adata.obsp.keys():
        compute_spatial_graph(adata, spatial_key, n_neighbors)
    if cluster_key not in adata.obs.keys():
        cluster_edges = cluster_cells_by_time(adata, t_key_true)
    metric.velocity_acc = velocity_accuracy(adata, v_key_pred, v_key_true)
    metric.spatial_time_consist = spatial_time_consistency(adata, t_key_pred, spatial_graph_key)
    metric.spatial_velo_consist = spatial_velocity_consistency(adata, v_key_pred, spatial_graph_key)
    # sc.pp.pca(adata)
    scv.pp.neighbors(adata, use_rep="X_coord", n_neighbors=n_neighbors)
    scv.tl.velocity_graph(adata, vkey=v_key_pred)
    scv.tl.velocity_embedding(adata, basis="coord", vkey=v_key_pred)
    adata_non_nan = adata[~np.isnan(adata.obsm[f"{v_key_pred}_coord"])].copy()
    metric.velo_emb_na_ratio = 1 - adata_non_nan.shape[0] / adata.shape[0]
    metric.cbdir = gen_cross_boundary_correctness(adata, cluster_key, v_key_pred, cluster_edges, t_key_true, spatial_graph_key, x_emb="X_coord")
    return metric

class BaseEvaluator:
    def __init__(self, 
                 cluster_edges=None,
                 cluster_key="t_cluster",
                 t_key_true="true_t",
                 v_key_true="true_velocity", 
                 spatial_graph_key="spatial_graph", 
                 spatial_key="coord",
                 u_key_true="Mu",
                 s_key_true="Ms",
                 n_neighbors=30,
                 title=None):
        self.metric = VelocityMetric(title=title)
        self.spatial_key = spatial_key
        self.n_neighbors = n_neighbors
        self.cluster_edges = cluster_edges
        self.cluster_key = cluster_key
        self.t_key_true = t_key_true
        self.v_key_true = v_key_true
        self.spatial_graph_key = spatial_graph_key
        self.u_key_true = u_key_true
        self.s_key_true = s_key_true

    def compute_time_corr(self, adata, tkey):
        self.metric.time_corr_spearman = adata.obs[[tkey, self.t_key_true]].corr(method="spearman").loc[self.t_key_true, tkey]

    def compute_time_corr_cell_gene(self, adata, tkey):
        adata.obs[f'{tkey}_mean'] = np.mean(adata.layers[tkey], axis=1)
        self.metric.time_corr_spearman_cell_gene = adata.obs[[f'{tkey}_mean', self.t_key_true]].corr(method="spearman").loc[self.t_key_true, f'{tkey}_mean']

    def comput_spatial_time_consist(self, adata, tkey, t_key_cluster):
        if self.spatial_key not in adata.obsp.keys():
            compute_spatial_graph(adata, f"X_{self.spatial_key}", self.n_neighbors)
        if self.cluster_key not in adata.obs.keys():
            # TODO: change t_key_true
            self.cluster_edges = cluster_cells_by_time(adata, t_key_cluster)
        self.metric.spatial_time_consist = spatial_time_consistency(adata, tkey, self.spatial_graph_key)

    def compute_velocity_acc(self, adata, vkey):
        self.metric.velocity_acc = velocity_accuracy(adata, vkey, self.v_key_true)

    def compute_cbdir(self, adata, vkey, tkey):
        scv.pp.neighbors(adata, use_rep=self.spatial_key, n_neighbors=self.n_neighbors)
        scv.tl.velocity_graph(adata, vkey=vkey)
        scv.tl.velocity_embedding(adata, vkey=vkey, basis=self.spatial_key)
        # TODO: change t_key_true
        self.metric.cbdir = gen_cross_boundary_correctness(adata, self.cluster_key, vkey, self.cluster_edges, tkey, self.spatial_graph_key, x_emb=f"X_{self.spatial_key}")
    
    def compute_spatial_velo_consist(self, adata, vkey):
        self.metric.spatial_velo_consist = spatial_velocity_consistency(adata, vkey, self.spatial_graph_key)

    def compute_abundance_mse(self, adata, ukey, skey):
        self.metric.abundance_mse = abundance_mse_us(adata, ukey, self.u_key_true, skey, self.s_key_true)

    def evaluate(self, adata, runner:BaseRunner, vkey, tkey, ukey, skey):
        if runner.is_real == False:
            self.compute_time_corr(adata, tkey=tkey)
            if isinstance(runner, scVeloRunner):
                self.compute_time_corr_cell_gene(adata, tkey=runner.t_cell_gene_key)
        if f"X_{self.spatial_key}" in adata.obsm.keys():
            self.comput_spatial_time_consist(adata, tkey=tkey, t_key_cluster=tkey if runner.is_real else self.t_key_true)
        adata_non_nan = adata[~np.isnan(adata.layers[f"{vkey}"]).any(axis=1)].copy()
        self.metric.velo_na_ratio = 1 - adata_non_nan.shape[0] / adata.shape[0]
        adata_non_nan_no_zero = adata_non_nan[(adata_non_nan.layers[f"{vkey}"] != 0).any(axis=1)].copy()
        self.metric.velo_zero_ratio = (adata_non_nan.shape[0] - adata_non_nan_no_zero.shape[0]) / adata.shape[0]
        if runner.is_real == False:
            self.compute_velocity_acc(adata_non_nan_no_zero, vkey=vkey)
        if f"X_{self.spatial_key}" in adata.obsm.keys():
            self.compute_cbdir(adata_non_nan_no_zero, vkey=vkey, tkey=tkey if runner.is_real else self.t_key_true)
            self.compute_spatial_velo_consist(adata_non_nan_no_zero, vkey)
        if runner.is_real == False:
            if runner.reconstruct_u is not None and runner.reconstruct_s is not None:
                self.compute_abundance_mse(adata, ukey, skey)
        self.metric.train_time = runner.run_time
        self.metric.mem_usage = runner.max_mem_usage
        self.metric.gpu_mem_usage = runner.max_gpu_mem_usage

    def draw_spatial_velocity(self, adata, basis, vkey, color):
        # if f"X_{self.spatial_key}" in adata.obsm.keys():
        #     scv.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep=self.spatial_key)
        # else:
        #     scv.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep="pca")
        scv.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep=basis)
        scv.tl.velocity_graph(adata, vkey=vkey,n_jobs=20)
        scv.pl.velocity_embedding_stream(adata, basis=basis, vkey=vkey, color=color)


if __name__ == "__main__":
    import scanpy as sc
    adata = sc.read_h5ad("/HDD1/chensishuo/spvelo_data/simulate/bidirect/exp_noise/0.5/20240605_222510/data_simu.h5ad")
    adata.obs["pred_t"] = adata.obs["true_t"]
    adata.layers["velocity_pred"] = adata.layers["velocity"]
    # sc.pp.pca(adata)
    metric = evaluate_metrics(adata, spatial_key="X_coord")
    print(metric)
    print(metric.cbdir_edge)
    print(metric.to_series())
