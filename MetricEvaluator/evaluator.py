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
import matplotlib.pyplot as plt
import scvelo as scv
import warnings

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


def cross_boundary_correctness(
        adata,
        k_cluster,
        k_velocity,
        cluster_edges,
        spatial_graph_key=None,
        return_raw=False,
        x_emb="X_umap",
        gene_mask=None):
    """Cross-Boundary Direction Correctness Score (A->B)

    Args:
        adata (:class:`anndata.AnnData`):
            Anndata object.
        k_cluster (str):
            Key to the cluster column in adata.obs.
        k_velocity (str):
            Key to the velocity matrix in adata.obsm.
        cluster_edges (list[tuple[str]]):
            Pairs of clusters has transition direction A->B
        return_raw (bool, optional):
            Return aggregated or raw scores. Defaults to False.
        x_emb (str, optional):
            Key to x embedding for visualization or a count matrix in adata.layers.
            Defaults to "X_umap".
        gene_mask (:class:`numpy.ndarray`, optional):
            Boolean array to filter out non-velocity genes. Defaults to None.

    Returns:
        tuple:

            - dict: all_scores indexed by cluster_edges or mean scores indexed by cluster_edges

            - float: averaged score over all cells
    """
    scores = {}
    all_scores = {}
    x_emb_name = x_emb
    if x_emb in adata.obsm:
        x_emb = adata.obsm[x_emb]
        if x_emb_name == "X_umap":
            v_emb = adata.obsm['{}_umap'.format(k_velocity)]
        else:
            v_emb = adata.obsm[f'{k_velocity}_{x_emb_name[2:]}']
    else:
        x_emb = adata.layers[x_emb]
        v_emb = adata.layers[k_velocity]
        if gene_mask is None:
            gene_mask = ~np.isnan(v_emb[0])
        x_emb = x_emb[:, gene_mask]
        v_emb = v_emb[:, gene_mask]
    cell_labels = adata.obs[k_cluster].to_numpy()
    for u, v in cluster_edges:
        sel = cell_labels == u
        if spatial_graph_key is None:
            nbs = adata.uns['neighbors']['indices'][sel]  # [n * 30]
        else:
            nbs = _get_spatial_nbs(adata, spatial_graph_key, np.where(sel)[0])

        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, k_cluster), nbs)
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]

        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0:
                continue
            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.nanmean(dir_scores))
        if len(type_score) == 0:
            print(f'Warning: cell type transition pair ({u},{v}) does not exist in the KNN graph. Ignored.')
        else:
            scores[f'{u} -> {v}'] = np.nanmean(type_score)
            all_scores[f'{u} -> {v}'] = type_score

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()])


# modified: make tkey optional
# k-CBDir
def gen_cross_boundary_correctness(
    adata,
    k_cluster,
    k_velocity,
    cluster_edges,
    tkey=None,
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
        tkey (str, optional):
            Key to the cell time in adata.obs
            Defaults to None.
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
    if tkey is not None: # changed
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
        if tkey is not None: # changed
            t_points = t[sel]

        type_score = [[] for i in range(k_hop)]
        type_score_null = [[] for i in range(k_hop)]
        
        # changed
        zipped = zip(x_points, x_velocities, boundary_nodes, t_points, nbs) if tkey is not None else zip(x_points, x_velocities, boundary_nodes, nbs)
        for vals in zipped:
            if tkey is not None:
                x_pos, x_vel, nodes, t_i, all_nodes = vals
            else:
                x_pos, x_vel, nodes, all_nodes = vals
            if len(nodes) == 0:
                continue
            position_dif = x_emb[nodes] - x_pos
            if tkey is not None: # changed
                dt = t[nodes] - t_i
                dir_scores = _cos_sim_sample(x_vel, position_dif, dt)
            else:
                dir_scores = _cos_sim_sample(x_vel, position_dif)

            nodes_null = all_nodes if len(all_nodes) < n_prune else np.random.choice(all_nodes, n_prune)
            position_dif_null = x_emb[nodes_null] - x_pos
            # changed
            if tkey is not None:
                dt_null = t[nodes_null] - t_i
                dir_scores_null = _cos_sim_sample(x_vel, position_dif_null, dt_null)
            else:
                dir_scores_null = _cos_sim_sample(x_vel, position_dif_null)

            # save 1-hop results
            type_score[0].append(np.nanmean(dir_scores))
            type_score_null[0].append(np.nanmean(dir_scores_null))
            # deal with k-hop neighbors when k > 1
            for k in range(k_hop-1):
                nodes = adata.uns['neighbors']['indices'][nodes].flatten()  # [num (k-1)-hop neighbors * 30]
                nodes = keep_type(adata, nodes, v, k_cluster)
                nodes = np.unique(nodes)

                position_dif = x_emb[nodes] - x_pos
                # changed
                if tkey is not None:
                    dt = t[nodes] - t_i
                    dir_scores = _cos_sim_sample(x_vel, position_dif, dt)
                else:
                    dir_scores = _cos_sim_sample(x_vel, position_dif)

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
                    # changed
                    if tkey is not None:
                        dt_null = t[nodes_null] - t_i
                        dir_scores_null = _cos_sim_sample(x_vel, position_dif_null, dt_null)
                    else:
                        dir_scores_null = _cos_sim_sample(x_vel, position_dif_null)
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

def _encode_type(cell_types_raw):
    #######################################################################
    # Use integer to encode the cell types
    # Each cell type has one unique integer label.
    #######################################################################
    # Map cell types to integers
    label_dic = {}
    label_dic_rev = {}
    for i, type_ in enumerate(cell_types_raw):
        label_dic[type_] = i
        label_dic_rev[i] = type_

    return label_dic, label_dic_rev


def _edge2adj(cell_types, cluster_edges):
    label_dic, label_dic_rev = _encode_type(cell_types)
    adj_mtx = np.zeros((len(cell_types), len(cell_types)))
    for u, v in cluster_edges:
        i, j = label_dic[u], label_dic[v]
        adj_mtx[i, j] = 1
        # child of child
        child_v = np.where(adj_mtx[j] > 0)[0]
        if len(child_v) > 0:
            adj_mtx[i, child_v] = 1
        # parent of parent
        par_u = np.where(adj_mtx[:, i] > 0)[0]
        if len(par_u) > 0:
            adj_mtx[par_u, j] = 1
    return label_dic, label_dic_rev, adj_mtx


def time_score(adata, tkey, cluster_key, cluster_edges):
    """Time Accuracy Score.
    Defined as the average proportion of descendant cells that
    appear after their progenitor cells.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        tkey (str):
            Key for the inferred cell time.
        cluster_key (str):
            Key for cell type annotations.
        cluster_edges (_type_, optional):
            Pairs of clusters has transition direction A->B.

    Returns:
        tuple:

            - dict: Time Accuracy Score for each transitio pair.

            - float: Mean Time Accuracy Score.
    """
    # Compute time inference accuracy based on
    # progenitor-descendant pairs
    t = adata.obs[tkey].to_numpy()
    cell_labels = adata.obs[cluster_key]
    cell_types = np.unique(cell_labels)
    label_dic, label_dic_rev, adj_mtx = _edge2adj(cell_types, cluster_edges)
    cell_labels_int = np.array([label_dic[x] for x in cell_labels])
    tscore = {}
    for i in range(adj_mtx.shape[0]):
        children = np.where(adj_mtx[i] > 0)[0]
        for j in children:
            p = np.nanmean(t[cell_labels_int == i] <= t[cell_labels_int == j].reshape(-1, 1))
            tscore[f'{label_dic_rev[i]} -> {label_dic_rev[j]}'] = p
    tscore_out = {}  # only return directly connected cell types in cluster_edges
    for u, v in cluster_edges:
        if (u, v) in cluster_edges:
            tscore_out[f'{u} -> {v}'] = tscore[f'{u} -> {v}']
    return tscore_out, np.nanmean([sc for sc in tscore.values()])


def inner_cluster_coh(adata, k_cluster, k_velocity, gene_mask=None, return_raw=False):
    """In-Cluster Coherence.
    Measures the average consistency of RNA velocity in each distinct cell type.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        k_cluster (str):
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str):
            key to the velocity matrix in adata.obsm.
        gene_mask (:class:`numpy.ndarray`, optional):
            Boolean array to filter out genes. Defaults to None.
        return_raw (bool, optional):
            return aggregated or raw scores.. Defaults to False.

    Returns:
        tuple:

            - dict: all_scores indexed by cluster_edges mean scores indexed by cluster_edges

            - float: Average score over all cells.
    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}

    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        nbs = adata.uns['neighbors']['indices'][sel]
        same_cat_nodes = map(lambda nodes: keep_type(adata, nodes, cat, k_cluster), nbs)

        velocities = adata.layers[k_velocity]
        nan_mask = ~np.isnan(velocities[0]) if gene_mask is None else gene_mask
        velocities = velocities[:, nan_mask]

        cat_vels = velocities[sel]
        cat_score = [cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean()
                     for ith, nodes in enumerate(same_cat_nodes)
                     if len(nodes) > 0]
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)

    if return_raw:
        return all_scores
    return scores, np.mean([sc for sc in scores.values()])


def _pearson_corr(v, v_neighbor):
    return np.corrcoef(v, v_neighbor)[0, 1:]


def velocity_consistency(adata, vkey, gene_mask=None):
    """Velocity Consistency as reported in scVelo paper

    Args:
        adata (:class:`anndata.AnnData`):
            Anndata object.
        vkey (str):
            key to the velocity matrix in adata.obsm.
        gene_mask (:class:`numpy.ndarray`, optional):
            Boolean array to filter out genes. Defaults to None.

    Returns:
        float: Average score over all cells.
    """
    nbs = adata.uns['neighbors']['indices']

    velocities = adata.layers[vkey]
    nan_mask = ~np.isnan(velocities[0]) if gene_mask is None else gene_mask
    
    velocities = velocities[:, nan_mask]

    consistency_score = [_pearson_corr(velocities[ith], velocities[nbs[ith]]).mean()
                         for ith in range(adata.n_obs)]
    adata.obs[f'{vkey}_consistency'] = consistency_score
    # return np.nanmean(consistency_score)
    return consistency_score


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
            continue
        consistency_score.append(_pearson_corr(velocities[ith], velocities[nbs_i]).mean())
    # adata.obs[f'{vkey}_consistency'] = consistency_score
    # return np.nanmean(consistency_score)
    return consistency_score


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
    bins = np.linspace(adata.obs[t_key].min(), adata.obs[t_key].max(), n_bins + 1)
    bins[-1] = bins[-1] + 1
    labels = [str(i) for i in range(n_bins)]
    cluster_edges = [(str(i), str(i + 1)) for i in range(n_bins - 1)]
    adata.obs["t_cluster"] = pd.cut(adata.obs[t_key], bins=bins, labels=labels, right=False)
    return cluster_edges


def abundance_mse_us(adata: AnnData, ukey_pred, ukey_true, skey_pred, skey_true):
    u_mse = mean_squared_error(adata.layers[ukey_true], adata.layers[ukey_pred])
    s_mse = mean_squared_error(adata.layers[skey_true], adata.layers[skey_pred])
    return u_mse, s_mse

class DictProxy:
    def __init__(self, all_scores, avg_score):
        self.all_scores = all_scores
        self.avg_score = avg_score
    
    def __getitem__(self, key):
        return self.all_scores[key], self.avg_score[key]
    
    def __setitem__(self, key, value):
        self.all_scores[key], self.avg_score[key] = value

    def __delitem__(self, key):
        del self.all_scores[key]
        del self.avg_score[key]

class VelocityMetric:
    def __init__(self, 
                 gen_cbdir=None, 
                 cbdir=None,
                 time_score=None,
                 inner_cluster_coh=None,
                 velocity_consistency=None,
                 time_corr_spearman=None, 
                #  time_corr_spearman_cell_gene=None,
                 velocity_acc=None, 
                 spatial_time_consist=None, 
                 spatial_velo_consist=None,
                    velo_na_ratio=None,
                    velo_zero_ratio=None,
                 train_time=None,
                 cpu_mem_usage=None,
                 gpu_mem_usage=None,
                 title=None,
                #  abundance_mse=None
                 ):
        self.title = title if title is not None else "Metric"
        
        self.gen_cbdir_edge = {}
        self.gen_cbdir_mean = {}
        self.gen_cbdir = gen_cbdir

        # self.cbdir_edge = {}
        # self.cbdir_mean = {}
        self.cbdir = cbdir if cbdir is not None else {}

        self.time_score_mean = {}
        self.time_score_edge = {}
        self.time_score = time_score

        # self.inner_cluster_coh_mean = {}
        # self.inner_cluster_coh_cate = {}
        self.inner_cluster_coh = inner_cluster_coh if inner_cluster_coh is not None else {}

        self.velocity_consistency = velocity_consistency if velocity_consistency is not None else {}

        self.velocity_acc = velocity_acc
        self.time_corr_spearman = time_corr_spearman if time_corr_spearman is not None else {}
        # self.time_corr_spearman_cell_gene = time_corr_spearman_cell_gene

        self.spatial_time_consist = spatial_time_consist if spatial_time_consist is not None else {}
        self.spatial_velo_consist = spatial_velo_consist if spatial_velo_consist is not None else {}

        # self.abundance_mse_u = None
        # self.abundance_mse_s = None
        # self.abundance_mse = abundance_mse

        self.velo_na_ratio = velo_na_ratio if velo_na_ratio is not None else {}
        self.velo_zero_ratio = velo_zero_ratio if velo_zero_ratio is not None else {}
        # self.velo_aggr_na_ratio = None
        # self.velo_aggr_zero_ratio = None

        self.train_time = train_time
        self.cpu_mem_usage = cpu_mem_usage
        self.gpu_mem_usage = gpu_mem_usage
    
    @property
    def gen_cbdir(self):
        return DictProxy(self.gen_cbdir_edge, self.gen_cbdir_mean)

    @gen_cbdir.setter
    def gen_cbdir(self, value):
        if value is None:
            self.gen_cbdir_edge = {}
            self.gen_cbdir_mean = {}
        else:
            self.gen_cbdir_edge, self.gen_cbdir_mean = value

    # @property
    # def cbdir(self):
    #     return DictProxy(self.cbdir_edge, self.cbdir_mean)
    
    # @cbdir.setter
    # def cbdir(self, value):
    #     if value is None:
    #         self.cbdir_edge = {}
    #         self.cbdir_mean = {}
    #     else:
    #         self.cbdir_edge, self.cbdir_mean = value

    @property
    def time_score(self):
        return DictProxy(self.time_score_edge, self.time_score_mean)
    
    @time_score.setter
    def time_score(self, value):
        if value is None:
            self.time_score_edge = {}
            self.time_score_mean = {}
        else:
            self.time_score_edge, self.time_score_mean = value

    # @property
    # def inner_cluster_coh(self):
    #     return DictProxy(self.inner_cluster_coh_cate, self.inner_cluster_coh_mean)
    
    # @inner_cluster_coh.setter
    # def inner_cluster_coh(self, value):
    #     if value is None:
    #         self.inner_cluster_coh_cate = {}
    #         self.inner_cluster_coh_mean = {}
    #     else:
    #         self.inner_cluster_coh_cate, self.inner_cluster_coh_mean = value

    # @property
    # def abundance_mse(self):
    #     return self.abundance_mse_u, self.abundance_mse_s

    # @abundance_mse.setter
    # def abundance_mse(self, value):
    #     if value is None:
    #         self.abundance_mse_u = None
    #         self.abundance_mse_s = None
    #     else:
    #         self.abundance_mse_u, self.abundance_mse_s = value

    def __str__(self):
        table = PrettyTable(["metric", "value"], title=self.title)
        if isinstance(self.gen_cbdir_mean, dict):
            for key, value in self.gen_cbdir_mean.items():
                table.add_row([f"gen_cbdir_mean_{key}", str([round(i, 4) for i in value]) if value is not None else None])
        if isinstance(self.gen_cbdir_edge, dict):
            for key, value in self.gen_cbdir_edge.items():
                table.add_row([f"gen_cbdir_edge_{key}", str({k: str([round(i, 4) for i in v]) if v is not None else None for k, v in value.items()}) if value is not None else None])
        # table.add_row(["gen_cbdir_mean", str([round(i, 4) for i in self.gen_cbdir_mean]) if self.gen_cbdir_mean is not None else None])
        # table.add_row(["gen_cbdir_edge", str({k: str([round(i, 4) for i in v]) for k, v in self.gen_cbdir_edge.items()}) if self.gen_cbdir_edge is not None else None])
        # if isinstance(self.cbdir_mean, dict):
        #     for key, value in self.cbdir_mean.items():
        #         table.add_row([f"cbdir_mean_{key}", round(value, 4) if value is not None else None])
        # if isinstance(self.cbdir_edge, dict):
        #     for key, value in self.cbdir_edge.items():
        #         table.add_row([f"cbdir_edge_{key}", str({k: round(v, 4) for k, v in value.items()}) if value is not None else None])
        if isinstance(self.cbdir, dict):
            for key, value in self.cbdir.items():
                table.add_row([f"cbdir_{key}", str({k: round(np.nanmean(v), 4) for k, v in value.items()}) if value is not None else None])
        # table.add_row(["cbdir_mean", round(self.cbdir_mean, 4) if self.cbdir_mean is not None else None])
        # table.add_row(["cbdir_edge", str({k: round(v, 4) for k, v in self.cbdir_edge.items()}) if self.cbdir_edge is not None else None])
        if isinstance(self.time_score_mean, dict):
            for key, value in self.time_score_mean.items():
                table.add_row([f"time_score_mean_{key}", round(value, 4) if value is not None else None])
        if isinstance(self.time_score_edge, dict):
            for key, value in self.time_score_edge.items():
                table.add_row([f"time_score_edge_{key}", str({k: round(v, 4) for k, v in value.items()}) if value is not None else None])
        # table.add_row(["time_score_mean", round(self.time_score_mean, 4) if self.time_score_mean is not None else None])
        # table.add_row(["time_score_edge", str({k: round(v, 4) for k, v in self.time_score_edge.items()}) if self.time_score_edge is not None else None])
        # if isinstance(self.inner_cluster_coh_mean, dict):
        #     for key, value in self.inner_cluster_coh_mean.items():
        #         table.add_row([f"inner_cluster_coh_mean_{key}", round(value, 4) if value is not None else None])
        # if isinstance(self.inner_cluster_coh_cate, dict):
        #     for key, value in self.inner_cluster_coh_cate.items():
        #         table.add_row([f"inner_cluster_coh_cate_{key}", str({k: round(v, 4) for k, v in value.items()}) if value is not None else None])
        if isinstance(self.inner_cluster_coh, dict):
            for key, value in self.inner_cluster_coh.items():
                table.add_row([f"inner_cluster_coh_{key}", str({k: round(np.nanmean(v), 4) for k, v in value.items()}) if value is not None else None])
        # table.add_row(["inner_cluster_coh_mean", round(self.inner_cluster_coh_mean, 4) if self.inner_cluster_coh_mean is not None else None])
        # table.add_row(["inner_cluster_coh_cate", str({k: round(v, 4) for k, v in self.inner_cluster_coh_cate.items()}) if self.inner_cluster_coh_cate is not None else None])
        if isinstance(self.velocity_consistency, dict):
            for key, value in self.velocity_consistency.items():
                table.add_row([f"velocity_consistency_{key}", round(np.nanmean(value), 4) if value is not None else None])
        # table.add_row(["velocity_consistency", round(self.velocity_consistency, 4) if self.velocity_consistency is not None else None])
        # table.add_row(["time_corr_spearman", self.time_corr_spearman])
        if isinstance(self.time_corr_spearman, dict):
            for key, value in self.time_corr_spearman.items():
                table.add_row([f"time_corr_spearman_{key}", round(value, 4)])
        # table.add_row(["time_corr_spearman_cell_gene", self.time_corr_spearman_cell_gene])
        table.add_row(["velocity_acc", round(self.velocity_acc, 4) if self.velocity_acc is not None else None])
        # table.add_row(["spatial_time_consist", round(self.spatial_time_consist, 4) if self.spatial_time_consist is not None else None])
        if isinstance(self.spatial_time_consist, dict):
            for key, value in self.spatial_time_consist.items():
                table.add_row([f"spatial_time_consist_{key}", round(value, 4)])
        if isinstance(self.spatial_velo_consist, dict):
            for key, value in self.spatial_velo_consist.items():
                table.add_row([f"spatial_velo_consist_{key}", round(np.nanmean(value), 4)])
        # table.add_row(["spatial_velo_consist", round(self.spatial_velo_consist, 4) if self.spatial_velo_consist is not None else None])
        if isinstance(self.velo_na_ratio, dict):
            for key, value in self.velo_na_ratio.items():
                table.add_row([f"velo_na_ratio_{key}", round(value, 4) if value is not None else None])
        # table.add_row(["velo_na_ratio", round(self.velo_na_ratio, 4) if self.velo_na_ratio is not None else None])
        if isinstance(self.velo_zero_ratio, dict):
            for key, value in self.velo_zero_ratio.items():
                table.add_row([f"velo_zero_ratio_{key}", round(value, 4) if value is not None else None])
        # table.add_row(["velo_zero_ratio", round(self.velo_zero_ratio, 4) if self.velo_zero_ratio is not None else None])
        # table.add_row(["abundance_mse_u", round(self.abundance_mse_u, 4) if self.abundance_mse_u is not None else None])
        # table.add_row(["abundance_mse_s", round(self.abundance_mse_s, 4) if self.abundance_mse_s is not None else None])
        table.add_row(["train_time", round(self.train_time, 2) if self.train_time is not None else None])
        table.add_row(["cpu_mem_usage", round(self.cpu_mem_usage, 2) if self.cpu_mem_usage is not None else None])
        table.add_row(["gpu_mem_usage", round(self.gpu_mem_usage) if self.gpu_mem_usage is not None else None])
        return table.__str__()

    def to_series(self):
        metric_dict = {
                    #     "gen_cbdir_mean": self.gen_cbdir_mean,
                    #    "gen_cbdir_edge": self.gen_cbdir_edge,
                    #     "cbdir_mean": self.cbdir_mean,
                    #     "cbdir_edge": self.cbdir_edge,
                    #    "time_score_mean": self.time_score_mean,
                    #    "time_score_edge": self.time_score_edge,
                    #    "inner_cluster_coh_mean": self.inner_cluster_coh_mean,
                    #    "inner_cluster_coh_cate": self.inner_cluster_coh_cate,
                    #    "velocity_consistency": self.velocity_consistency,
                    #    "time_corr_spearman": self.time_corr_spearman,
                    #    "time_corr_spearman_cell_gene": self.time_corr_spearman_cell_gene,
                       "velocity_acc": self.velocity_acc,
                    #    "spatial_time_consist": self.spatial_time_consist,
                    #    "spatial_velo_consist": self.spatial_velo_consist,
                    #    "velo_na_ratio": self.velo_na_ratio,
                    #    "velo_zero_ratio": self.velo_zero_ratio,
                    #    "abundance_mse_u": self.abundance_mse_u,
                    #    "abundance_mse_s": self.abundance_mse_s,
                        "train_time": self.train_time,
                       "cpu_mem_usage": self.cpu_mem_usage,
                       "gpu_mem_usage": self.gpu_mem_usage
                       }
        if isinstance(self.gen_cbdir_mean, dict):
            for key, value in self.gen_cbdir_mean.items():
                metric_dict[f"gen_cbdir_mean_{key}"] = value
        if isinstance(self.gen_cbdir_edge, dict):
            for key, value in self.gen_cbdir_edge.items():
                for k, v in value.items():
                    metric_dict[f"gen_cbdir_edge_{key}_{k}"] = v
        # if isinstance(self.cbdir_mean, dict):
        #     for key, value in self.cbdir_mean.items():
        #         metric_dict[f"cbdir_mean_{key}"] = value
        # if isinstance(self.cbdir_edge, dict):
        #     for key, value in self.cbdir_edge.items():
        #         for k, v in value.items():
        #             metric_dict[f"cbdir_edge_{key}_{k}"] = v
        if isinstance(self.cbdir, dict):
            for key, value in self.cbdir.items():
                for k, v in value.items():
                    metric_dict[f"cbdir_{key}_{k}"] = np.nanmean(v)
        if isinstance(self.time_score_mean, dict):
            for key, value in self.time_score_mean.items():
                metric_dict[f"time_score_mean_{key}"] = value
        if isinstance(self.time_score_edge, dict):
            for key, value in self.time_score_edge.items():
                for k, v in value.items():
                    metric_dict[f"time_score_edge_{key}_{k}"] = v
        # if isinstance(self.inner_cluster_coh_mean, dict):
        #     for key, value in self.inner_cluster_coh_mean.items():
        #         metric_dict[f"inner_cluster_coh_mean_{key}"] = value
        # if isinstance(self.inner_cluster_coh_cate, dict):
        #     for key, value in self.inner_cluster_coh_cate.items():
        #         for k, v in value.items():
        #             metric_dict[f"inner_cluster_coh_cate_{key}_{k}"] = v
        if isinstance(self.inner_cluster_coh, dict):
            for key, value in self.inner_cluster_coh.items():
                for k, v in value.items():
                    metric_dict[f"inner_cluster_coh_{key}_{k}"] = np.nanmean(v)
        if isinstance(self.velocity_consistency, dict):
            for key, value in self.velocity_consistency.items():
                metric_dict[f"velocity_consistency_{key}"] = np.nanmean(value)
        if isinstance(self.time_corr_spearman, dict):
            for key, value in self.time_corr_spearman.items():
                metric_dict[f"time_corr_spearman_{key}"] = value
        if isinstance(self.spatial_time_consist, dict):
            for key, value in self.spatial_time_consist.items():
                metric_dict[f"spatial_time_consist_{key}"] = value
        if isinstance(self.spatial_velo_consist, dict):
            for key, value in self.spatial_velo_consist.items():
                metric_dict[f"spatial_velo_consist_{key}"] = np.nanmean(value)
        if isinstance(self.velo_na_ratio, dict):
            for key, value in self.velo_na_ratio.items():
                metric_dict[f"velo_na_ratio_{key}"] = value
        if isinstance(self.velo_zero_ratio, dict):
            for key, value in self.velo_zero_ratio.items():
                metric_dict[f"velo_zero_ratio_{key}"] = value
        return pd.Series(metric_dict)

# NOTE: unused
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
    metric.gen_cbdir = gen_cross_boundary_correctness(adata, cluster_key, v_key_pred, cluster_edges, t_key_true, spatial_graph_key, x_emb="X_coord")
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
        self.transition_confidence = None
        # self.is_neighbors_recomputed = False

    def compute_time_corr(self, adata, tkey):
        # if self.metric.time_corr_spearman is None:
        #     self.metric.time_corr_spearman = {}
        self.metric.time_corr_spearman[tkey] = adata.obs[[tkey, self.t_key_true]].corr(method="spearman").loc[self.t_key_true, tkey]

    # def compute_time_corr_cell_gene(self, adata, tkey):
    #     adata.obs[f'{tkey}_mean'] = np.mean(adata.layers[tkey], axis=1)
    #     self.metric.time_corr_spearman_cell_gene = adata.obs[[f'{tkey}_mean', self.t_key_true]].corr(method="spearman").loc[self.t_key_true, f'{tkey}_mean']

    def compute_velocity_acc(self, adata, vkey):
        self.metric.velocity_acc = velocity_accuracy(adata, vkey, self.v_key_true)

    def compute_spatial_time_consist(self, adata, tkey):
        if self.spatial_graph_key not in adata.obsp.keys():
            compute_spatial_graph(adata, f"X_{self.spatial_key}", self.n_neighbors)
        # if self.metric.spatial_time_consist is None:
        #     self.metric.spatial_time_consist = {}
        self.metric.spatial_time_consist[tkey] = spatial_time_consistency(adata, tkey, self.spatial_graph_key)

    def compute_spatial_velo_consist(self, adata, vkey):
        if self.spatial_graph_key not in adata.obsp.keys():
            compute_spatial_graph(adata, f"X_{self.spatial_key}", self.n_neighbors)
        # if self.metric.spatial_velo_consist is None:
        #     self.metric.spatial_velo_consist = {}
        self.metric.spatial_velo_consist[vkey] = spatial_velocity_consistency(adata, vkey, self.spatial_graph_key)

    def compute_gen_cbdir(self, adata, vkey, basis, tkey=None): # TODO: whether to use basis to compute neighbors
        
        if self.spatial_graph_key not in adata.obsp.keys() and f"X_{self.spatial_key}" in adata.obsm.keys():
            compute_spatial_graph(adata, f"X_{self.spatial_key}", self.n_neighbors)
        elif f"X_{self.spatial_key}" not in adata.obsm.keys():
            self.spatial_graph_key = None
        
        # if self.cluster_edges == None and self.t_key_true in adata.obs.keys():
        #     self.cluster_key = "t_cluster"
        #     self.cluster_edges = cluster_cells_by_time(adata, self.t_key_true)
        # else:
        #     raise KeyError("cluster_edges or t_key_true not found in adata.obs")
        # if self.is_neighbors_recomputed == False: # FIXME: adata.uns['neighbors']['indices']
        #     scv.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep=basis)
        #     self.is_neighbors_recomputed = True
        if 'indices' not in adata.uns['neighbors']:
            scv.pp.neighbors(adata, n_neighbors=self.n_neighbors)
        if f"{vkey}_graph" not in adata.uns:
            scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=20)
        if f"{vkey}_{basis}" not in adata.obsm.keys():
            scv.tl.velocity_embedding(adata, vkey=vkey, basis=basis)
        cell_types = np.unique(adata.obs[self.cluster_key])
        valid_edges = [(u, v) for u, v in self.cluster_edges if u in cell_types and v in cell_types]
        self.metric.gen_cbdir[vkey] = gen_cross_boundary_correctness(adata, self.cluster_key, vkey, valid_edges, tkey, x_emb=f"X_{basis}") # TODO: spatial_graph
    
    def compute_cbdir(self, adata, vkey, basis): # TODO: whether to use basis to compute neighbors

        if self.spatial_graph_key not in adata.obsp.keys() and f"X_{self.spatial_key}" in adata.obsm.keys():
            compute_spatial_graph(adata, f"X_{self.spatial_key}", self.n_neighbors)
        elif f"X_{self.spatial_key}" not in adata.obsm.keys():
            self.spatial_graph_key = None

        # if self.cluster_key == "t_cluster" and self.cluster_key not in adata.obs.keys() and self.t_key_true in adata.obs.keys():
        #     self.cluster_edges = cluster_cells_by_time(adata, self.t_key_true)
        # elif self.cluster_key not in adata.obs.keys():
        #     raise KeyError("cluster_key or t_key_true not found in adata.obs")
        # if self.is_neighbors_recomputed == False: # FIXME: adata.uns['neighbors']['indices']
        #     scv.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep=basis)
        #     self.is_neighbors_recomputed = True
        if 'indices' not in adata.uns['neighbors']:
            scv.pp.neighbors(adata, n_neighbors=self.n_neighbors)
        if f"{vkey}_graph" not in adata.uns:
            scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=20)
        if f"{vkey}_{basis}" not in adata.obsm.keys():
            scv.tl.velocity_embedding(adata, vkey=vkey, basis=basis)
        cell_types = np.unique(adata.obs[self.cluster_key])
        valid_edges = [(u, v) for u, v in self.cluster_edges if u in cell_types and v in cell_types]
        self.metric.cbdir[vkey] = cross_boundary_correctness(adata, self.cluster_key, vkey, valid_edges, return_raw=True, x_emb=f"X_{basis}") # TODO: spatial_graph

    def compute_time_score(self, adata, tkey):
        # if self.cluster_key == "t_cluster" and self.cluster_key not in adata.obs.keys() and self.t_key_true in adata.obs.keys():
        #     self.cluster_edges = cluster_cells_by_time(adata, self.t_key_true)
        # elif self.cluster_key not in adata.obs.keys():
        #     raise KeyError("cluster_key or t_key_true not found in adata.obs")
        cell_types = np.unique(adata.obs[self.cluster_key])
        valid_edges = [(u, v) for u, v in self.cluster_edges if u in cell_types and v in cell_types]
        self.metric.time_score[tkey] = time_score(adata, tkey, self.cluster_key, valid_edges)

    def compute_inner_cluster_coh(self, adata, vkey): # TODO: whether to use basis to compute neighbors
        # if self.is_neighbors_recomputed == False: # FIXME: adata.uns['neighbors']['indices']
        #     scv.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep=basis)
        #     self.is_neighbors_recomputed = True
        if 'indices' not in adata.uns['neighbors']:
            scv.pp.neighbors(adata, n_neighbors=self.n_neighbors)
        self.metric.inner_cluster_coh[vkey] = inner_cluster_coh(adata, self.cluster_key, vkey, return_raw=True)

    def compute_velocity_consistency(self, adata, vkey): # TODO: whether to use basis to compute neighbors
        # if self.is_neighbors_recomputed == False: # FIXME: adata.uns['neighbors']['indices']
        #     scv.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep=basis)
        #     self.is_neighbors_recomputed = True
        if 'indices' not in adata.uns['neighbors']:
            scv.pp.neighbors(adata, n_neighbors=self.n_neighbors)
        # if self.metric.velocity_consistency is None:
        #     self.metric.velocity_consistency = {}
        self.metric.velocity_consistency[vkey] = velocity_consistency(adata, vkey)

    # def compute_abundance_mse(self, adata, ukey, skey):
    #     self.metric.abundance_mse = abundance_mse_us(adata, ukey, self.u_key_true, skey, self.s_key_true)

    def evaluate(self, adata, runner:BaseRunner, vkey, basis, tkey=None, t_key_mean=None, adata_aggr=None, vkey_aggr='vj', use_rep=False):
        # NOTE: temporary solution
        if use_rep:
            scv.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep=basis)
        if self.cluster_edges == None and self.t_key_true in adata.obs.keys():
            self.cluster_key = "t_cluster"
            self.cluster_edges = cluster_cells_by_time(adata, self.t_key_true)
        
        adata_no_nan = adata[~np.isnan(adata.layers[vkey]).any(axis=1)].copy()
        self.metric.velo_na_ratio[vkey] = 1 - adata_no_nan.shape[0] / adata.shape[0]
        adata_no_nan_no_zero = adata_no_nan[(adata_no_nan.layers[vkey] != 0).any(axis=1)].copy()
        self.metric.velo_zero_ratio[vkey] = (adata_no_nan.shape[0] - adata_no_nan_no_zero.shape[0]) / adata.shape[0]

        if runner.is_real == False: # and runner.latent_time is not None:
            if tkey is not None:
                self.compute_time_corr(adata, tkey=tkey)
            if t_key_mean is not None:
                self.compute_time_corr(adata, tkey=t_key_mean)
            self.compute_velocity_acc(adata_no_nan_no_zero, vkey=vkey)
            # if isinstance(runner, scVeloRunner):
            #     self.compute_time_corr_cell_gene(adata, tkey=runner.t_cell_gene_key)
        if self.spatial_key is not None and f"X_{self.spatial_key}" in adata.obsm.keys(): # and runner.latent_time is not None:
            if tkey is not None:
                self.compute_spatial_time_consist(adata, tkey=tkey)
            if t_key_mean is not None:
                self.compute_spatial_time_consist(adata, tkey=t_key_mean)
            self.compute_spatial_velo_consist(adata_no_nan_no_zero, vkey)

        self.compute_velocity_consistency(adata_no_nan_no_zero, vkey)
            
        self.compute_gen_cbdir(adata_no_nan_no_zero, vkey=vkey, basis=basis) # TODO: tkey
        self.compute_cbdir(adata_no_nan_no_zero, vkey=vkey, basis=basis)
        if tkey is not None:
            self.compute_time_score(adata, tkey=tkey)
        if t_key_mean is not None:
            self.compute_time_score(adata, tkey=t_key_mean)

        self.compute_inner_cluster_coh(adata_no_nan_no_zero, vkey)

        self.metric.train_time = runner.run_time
        self.metric.cpu_mem_usage = runner.max_cpu_mem_usage
        self.metric.gpu_mem_usage = runner.max_gpu_mem_usage

        if adata_aggr is not None:
            # self.is_neighbors_recomputed = False
            if f'X_{basis}' not in adata_aggr.obsm.keys():
                adata_aggr.obsm[f'X_{basis}'] = adata.obsm[f'X_{basis}']
            if f'X_{self.spatial_key}' not in adata_aggr.obsm.keys() and self.spatial_key is not None and f"X_{self.spatial_key}" in adata.obsm.keys() and self.spatial_key != basis:
                adata_aggr.obsm[f'X_{self.spatial_key}'] = adata.obsm[f'X_{self.spatial_key}']
            if self.cluster_key not in adata_aggr.obs.keys():
                adata_aggr.obs[self.cluster_key] = adata.obs[self.cluster_key].values
            if self.t_key_true in adata.obs.keys() and self.t_key_true not in adata_aggr.obs.keys():
                adata_aggr.obs[self.t_key_true] = adata.obs[self.t_key_true].values
            
            adata_aggr_no_nan = adata_aggr[~np.isnan(adata_aggr.layers[vkey_aggr]).any(axis=1)].copy()
            self.metric.velo_na_ratio[vkey_aggr] = 1 - adata_aggr_no_nan.shape[0] / adata_aggr.shape[0]
            adata_aggr_no_nan_no_zero = adata_aggr_no_nan[(adata_aggr_no_nan.layers[vkey_aggr] != 0).any(axis=1)].copy()
            self.metric.velo_zero_ratio[vkey_aggr] = (adata_aggr_no_nan.shape[0] - adata_aggr_no_nan_no_zero.shape[0]) / adata_aggr.shape[0]

            if self.spatial_key is not None and f"X_{self.spatial_key}" in adata.obsm.keys():
                self.compute_spatial_velo_consist(adata_aggr_no_nan_no_zero, vkey_aggr)
            self.compute_velocity_consistency(adata_aggr_no_nan_no_zero, vkey_aggr)
            self.compute_gen_cbdir(adata_aggr_no_nan_no_zero, vkey_aggr, basis)
            self.compute_cbdir(adata_aggr_no_nan_no_zero, vkey_aggr, basis)
            self.compute_inner_cluster_coh(adata_aggr_no_nan_no_zero, vkey_aggr)

    # NOTE: not used
    def compute_and_plot_PAGA_graph(self, adata, vkey, basis):
        if f"{vkey}_graph" not in adata.uns:
            scv.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep=basis)
            scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=20)
        adata.uns['neighbors']['distances'] = adata.obsp['distances']
        adata.uns['neighbors']['connectivities'] = adata.obsp['connectivities']
        scv.tl.paga(adata, self.cluster_key, vkey)
        self.transition_confidence = scv.get_df(adata, 'paga/transitions_confidence', precision=2).T
        scv.pl.paga(adata, basis=basis, vkey=vkey, size=50, alpha=.1,
            min_edge_width=2, node_size_scale=1.5, color=self.cluster_key)
        plt.tight_layout()

    def plot_velocity_stream(self, adata, basis, vkey, save_dir, adata_aggr=None, vkey_aggr='vj'):
        # if f"X_{self.spatial_key}" in adata.obsm.keys():
        #     scv.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep=self.spatial_key)
        # else:
        #     scv.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep="pca")
        if f"{vkey}_graph" not in adata.uns:
            # scv.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep=basis) # FIXME:
            scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=20)
        scv.pl.velocity_embedding_stream(adata, basis=basis, vkey=vkey, color=self.cluster_key if self.cluster_key != 't_cluster' else self.t_key_true,
                                        legend_loc="lower right", legend_fontsize=8, color_map='viridis' if self.cluster_key=='t_cluster' else None)
        if self.cluster_key != 't_cluster':
            plt.tight_layout()
        plt.savefig(f"{save_dir}/{vkey}_{basis}.svg", format='svg', dpi=300)
        if adata_aggr is not None:
            if f'X_{basis}' not in adata_aggr.obsm.keys():
                adata_aggr.obsm[f'X_{basis}'] = adata.obsm[f'X_{basis}']
            if f'X_{self.spatial_key}' not in adata_aggr.obsm.keys() and self.spatial_key is not None and f"X_{self.spatial_key}" in adata.obsm.keys() and self.spatial_key != basis:
                adata_aggr.obsm[f'X_{self.spatial_key}'] = adata.obsm[f'X_{self.spatial_key}']
            if self.cluster_key not in adata_aggr.obs.keys():
                adata_aggr.obs[self.cluster_key] = adata.obs[self.cluster_key]
            if self.t_key_true in adata.obs.keys() and self.t_key_true not in adata_aggr.obs.keys():
                adata_aggr.obs[self.t_key_true] = adata.obs[self.t_key_true]
            if f"{vkey_aggr}_graph" not in adata_aggr.uns:
                # scv.pp.neighbors(adata_aggr, n_neighbors=self.n_neighbors, use_rep=basis) # FIXME:
                scv.tl.velocity_graph(adata_aggr, vkey=vkey_aggr, n_jobs=20)
            scv.pl.velocity_embedding_stream(adata_aggr, basis=basis, vkey=vkey_aggr, color=self.cluster_key if self.cluster_key != 't_cluster' else self.t_key_true,
                                             legend_loc="lower right", legend_fontsize=8, color_map='viridis' if self.cluster_key=='t_cluster' else None)
            if self.cluster_key != 't_cluster':
                plt.tight_layout()
            plt.savefig(f"{save_dir}/{vkey_aggr}_{basis}.svg", format='svg', dpi=300)

    def plot_time(self, adata, basis, tkey, save_dir, tkey_mean=None):
        scv.pl.scatter(adata, basis=basis, color=tkey, color_map='viridis')
        plt.savefig(f"{save_dir}/{tkey}_{basis}.svg", format='svg', dpi=300)
        if tkey_mean is not None:
            scv.pl.scatter(adata, basis=basis, color=tkey_mean, color_map='viridis')
            plt.savefig(f"{save_dir}/{tkey_mean}_{basis}.svg", format='svg', dpi=300)


if __name__ == "__main__":
    import scanpy as sc
    adata = sc.read_h5ad("/HDD1/chensishuo/spvelo_data/simulate/bidirect/exp_noise/0.5/20240605_222510/data_simu.h5ad")
    adata.obs["pred_t"] = adata.obs["true_t"]
    adata.layers["velocity_pred"] = adata.layers["velocity"]
    # sc.pp.pca(adata)
    metric = evaluate_metrics(adata, spatial_key="X_coord")
    print(metric)
    print(metric.gen_cbdir_edge)
    print(metric.to_series())
