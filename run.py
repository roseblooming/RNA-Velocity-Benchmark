import sys
sys.path.append("models")

import scanpy as sc
import scvelo as scv
# from Runner.stt_runner import STT_Runner
# from Runner.deepvelo_cui_runner import DeepVeloCuiRunner
from Runner.BaseRunner import BaseRunner
from Runner.unitvelo_runner import UniTVeloRunner
from Runner.velovi_runner import veloVI_Runner
from Runner.velovae_runner import VeloVAE_Runner
from Runner.topovelo_runner import TopoVeloRunner
from Runner.scvelo_runner import scVeloRunner
from MetricEvaluator.evaluator import BaseEvaluator
from sklearn.preprocessing import MinMaxScaler
import argparse
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata_path", type=str, required=True, help="path to h5ad")
    parser.add_argument("--save_dir", type=str, default=None, help="save dir")
    parser.add_argument("--device", type=int, default=None, help="device to train")
    parser.add_argument("--model", type=str, required=True, choices=["scvelo", "topovelo", "velovae", "velovi", "unitvelo", "deepvelo_cui", "stt"], help="model to evaluate")
    parser.add_argument("--real_data", action="store_true", default=False, help="wheather to run on real data")
    parser.add_argument("--label", type=str, default=None, help="label or cluster key")
    parser.add_argument("--spatial_key", type=str, default=None, help="spatial key")
    parser.add_argument("--basis", type=str, nargs='+', required=True, help="basis to evaluate and plot")
    parser.add_argument("--cluster_edges_path", type=str, default=None, help="path to cluster edges")

    parser.add_argument("--redo", action="store_true", default=False, help="re-evaluate results without re-training")
    parser.add_argument("--do_not_plot", action="store_true", default=False, help="do not plot")
    parser.add_argument("--do_not_eval", action="store_true", default=False, help="do not evaluate")

    parser.add_argument("--n_states", type=int, default=None, help="number of states (only for stt)")
    parser.add_argument("--adata_aggr_path", type=str, default=None, help="path to aggregated adata (only for stt when redo is True)")

    parser.add_argument("--varn_unique", action="store_true", default=False, help="make var names unique (required for hsc)")
    parser.add_argument("--dimred", type=str, default=None, help="add dimred basis (adata.obsm['X_dimred'] = adata.obsm['dimred'])")
    parser.add_argument("--del_ori_nbs", action="store_true", default=False, help="delete original neighbors")
    parser.add_argument("--pp_choice", type=int, default=0, choices=[0, 1, 2], help="0: default, 1: only moments, 2: no preprocess")
    parser.add_argument("--leiden_resolution", type=float, default=None, help="resolution for leiden clustering (only for simu data, 0.01 for bidirectional, 0.1 for radial)")
    parser.add_argument("--use_rep_for_nbs", action="store_true", default=False, help="whether to use basis for re-computing neighbors")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    adata = sc.read_h5ad(args.adata_path)

    if args.varn_unique:
        adata.var_names_make_unique()
    if args.dimred is not None:
        adata.obsm[f"X_{args.dimred}"] = adata.obsm[args.dimred]
    if args.del_ori_nbs and 'neighbors' in adata.uns.keys():
        del adata.uns['neighbors']

    if args.leiden_resolution is not None:
        adata_orig = adata.copy()
        adata_orig.X = adata_orig.layers['true_spliced'].copy()
        adata_orig.layers['spliced'] = adata_orig.layers['true_spliced'].copy()
        adata_orig.layers['unspliced'] = adata_orig.layers['true_unspliced'].copy()
        sc.tl.pca(adata_orig, n_comps=50)
        sc.pp.neighbors(adata_orig, n_neighbors=50, n_pcs=50)
        sc.tl.leiden(adata_orig, resolution=args.leiden_resolution)
        adata.obs['leiden'] = adata_orig.obs['leiden']
        adata.obsm['X_pca_orig'] = adata_orig.obsm['X_pca']
    
    if args.cluster_edges_path is not None:
        with open(args.cluster_edges_path, "rb") as f:
            cluster_edges = pickle.load(f)
    else:
        cluster_edges = None
    
    # adata.obsm["X_pca_backup"] = adata.obsm["X_pca"].copy()
    # adata.obsm["X_umap_backup"] = adata.obsm["X_umap"].copy()
    if args.save_dir is None:
        args.save_dir = time.strftime(f"logs/{args.model}/%Y%m%d-%H%M%S", time.localtime())
    os.makedirs(args.save_dir, exist_ok=True)
    pd.Series(vars(args)).to_csv(f"{args.save_dir}/config.csv")
    if args.model == "scvelo":
        runner = scVeloRunner("dynamical", adata, args.real_data, args.pp_choice, args.save_dir)
    elif args.model == "topovelo":
        runner = TopoVeloRunner(adata, args.spatial_key, args.real_data, args.pp_choice, device=args.device, save_dir=args.save_dir)
    elif args.model == "velovae":
        runner = VeloVAE_Runner(adata, args.real_data, args.pp_choice, device=args.device, save_dir=args.save_dir)
    elif args.model == "velovi":
        runner = veloVI_Runner(adata, args.real_data, args.pp_choice, device=args.device, save_dir=args.save_dir)
    elif args.model == "unitvelo":
        runner = UniTVeloRunner(adata, args.real_data, args.pp_choice, args.device, args.save_dir, label=args.label)
    elif args.model == "deepvelo_cui":
        # runner = DeepVeloCuiRunner(adata, args.real_data, args.pp_choice, device=args.device, save_dir=args.save_dir)
        runner = BaseRunner(args.model, adata, args.real_data, args.pp_choice, args.save_dir)
    # elif args.model == "stt":
    #     runner = STT_Runner(adata, args.real_data, args.pp_choice, args.label, args.spatial_key, args.n_states, args.save_dir)

    if args.redo == False:
        runner.run()
        if args.model == "velovi" and args.real_data == True: # TODO: solve reference problem
            adata = runner.adata
    elif args.model == "stt":
        runner.adata_aggr = sc.read_h5ad(args.adata_aggr_path)
    if args.model == "scvelo" and "velocity_genes" in adata.var:
        adata = adata[:, adata.var["velocity_genes"]].copy()

    if args.model != "stt":
        scaler = MinMaxScaler()
        adata.obs[f'{runner.tkey}_norm'] = scaler.fit_transform(adata.obs[[runner.tkey]])
    if args.model == "scvelo":
        adata.obs[f'{runner.t_cell_gene_key}_mean'] = adata.layers[runner.t_cell_gene_key].mean(axis=1)
        adata.obs[f'{runner.t_cell_gene_key}_norm'] = scaler.fit_transform(adata.obs[[f'{runner.t_cell_gene_key}_mean']])
    
    evaluator = BaseEvaluator(cluster_edges=cluster_edges, cluster_key=args.label if args.real_data else 't_cluster', spatial_key=args.spatial_key)
    for basis in args.basis:
        if basis == 'umap' and 'X_umap' not in adata.obsm.keys():
            sc.tl.umap(adata)
        if not args.do_not_eval:
            evaluator.evaluate(adata, runner, runner.vkey, basis, f'{runner.tkey}_norm' if args.model != "stt" else None, f'{runner.t_cell_gene_key}_norm' if args.model == "scvelo" else None, runner.adata_aggr if args.model == "stt" else None)
            metric = evaluator.metric
            print(metric)
            metric_series = metric.to_series()
            metric_series.to_csv(path_or_buf=f"{args.save_dir}/metric_{basis}.csv" if args.redo == False else f"{args.save_dir}/metric_{basis}_redo.csv")

            cbdir = metric.cbdir
            if isinstance(cbdir, dict):
                for key, value in cbdir.items():
                    with open(f"{args.save_dir}/cbdir_{basis}_{key}.pkl", "wb") as f:
                        pickle.dump(value, f)
            inner_cluster_coh = metric.inner_cluster_coh
            if isinstance(inner_cluster_coh, dict):
                for key, value in inner_cluster_coh.items():
                    with open(f"{args.save_dir}/inner_cluster_coh_{basis}_{key}.pkl", "wb") as f:
                        pickle.dump(value, f)
            velocity_consistency = metric.velocity_consistency
            if isinstance(velocity_consistency, dict):
                for key, value in velocity_consistency.items():
                    with open(f"{args.save_dir}/velocity_consistency_{basis}_{key}.pkl", "wb") as f:
                        pickle.dump(value, f)
            spatial_velo_consist = metric.spatial_velo_consist
            if isinstance(spatial_velo_consist, dict):
                for key, value in spatial_velo_consist.items():
                    with open(f"{args.save_dir}/spatial_velo_consist_{basis}_{key}.pkl", "wb") as f:
                        pickle.dump(value, f)

        # color_key = None
        # if args.real_data == False:
        #     color_key = evaluator.t_key_true
        ###### 保留 ######
        # if args.model != "stt":
        #     color_key = f'{runner.tkey}_norm'
        # else:
        #     color_key = None
        ###### 保留 ######
        # if args.spatial_key is not None and f"X_{args.spatial_key}" in adata.obsm.keys():
        #     plot(args.label, args.save_dir, adata, runner, evaluator, color_key, args.spatial_key)
        # if args.basis == "pca":
        #     sc.pp.pca(adata)
        #     plot(args.label, args.save_dir, adata, runner, evaluator, color_key, args.basis)
        # if args.basis != args.spatial_key and f"X_{args.basis}" in adata.obsm.keys():
        ###### 保留 ######
        if not args.do_not_plot:
            # plot(args.label, args.save_dir, adata, runner, evaluator, color_key, basis)
            evaluator.plot_velocity_stream(adata, basis, runner.vkey, args.save_dir, runner.adata_aggr if args.model == "stt" else None)
            # plt.savefig(f"{args.save_dir}/velocity_{basis}.svg", format='svg', dpi=300)
            if args.model != "stt":
                evaluator.plot_time(adata, basis, f'{runner.tkey}_norm' if not args.do_not_eval else runner.tkey, args.save_dir, f'{runner.t_cell_gene_key}_norm' if args.model == "scvelo" else None)
                # plt.savefig(f"{args.save_dir}/time_{basis}.svg", format='svg', dpi=300)

        # evaluator.compute_and_plot_PAGA_graph(adata, runner.vkey, f'{runner.tkey}_norm', basis)
        # plt.savefig(f"{args.save_dir}/PAGA_{basis}.svg", format='svg', dpi=300)
        # evaluator.transition_confidence.to_csv(f"{args.save_dir}/transition_confidence_{basis}.csv")
        ###### 保留 ######

    # sc.pp.pca(adata)
    # adata.obsm["X_pca"] = adata.obsm["X_pca_backup"].copy()
    # adata.obsm["X_umap"] = adata.obsm["X_umap_ori"].copy()
    # plot(args.label, args.save_dir, adata, runner, evaluator, color_key, "pca")

# NOTE: not used
def plot(label, save_dir, adata, runner:BaseRunner, evaluator:BaseEvaluator, color_key, basis):
    if label is not None:
        evaluator.plot_velocity_stream(adata, basis=basis, vkey=runner.vkey, color=label)
        plt.savefig(f"{save_dir}/velocity_{basis}_label.svg", format='svg', dpi=300)
        if color_key is not None:
            scv.pl.scatter(adata, basis=basis, color=color_key, color_map="viridis")
            plt.savefig(f"{save_dir}/{basis}_time.svg", format='svg', dpi=300)
    elif color_key is not None:
        evaluator.plot_velocity_stream(adata, basis=basis, vkey=runner.vkey, color=color_key, color_map="viridis")
        plt.savefig(f"{save_dir}/velocity_{basis}_time.svg", format='svg', dpi=300)

    # sc.tl.umap(adata)
    # adata.obsm["X_umap"] = adata.obsm["X_umap_ori"].copy()
    # basis = "umap"
    # if args.label is not None:
    #     evaluator.draw_spatial_velocity(adata, basis=basis, vkey=runner.vkey, color=args.label)
    #     plt.savefig(f"{args.save_dir}/velocity_{basis}_label.svg", format='svg', dpi=300)
    #     plt.savefig(f"{args.save_dir}/velocity_{basis}_label.png", format='png', dpi=300)
    #     scv.pl.scatter(adata, basis=basis, color=color_key)
    #     plt.savefig(f"{args.save_dir}/{basis}_time.svg", format='svg', dpi=300)
    #     plt.savefig(f"{args.save_dir}/{basis}_time.png", format='png', dpi=300)
    # else:
    #     evaluator.draw_spatial_velocity(adata, basis=basis, vkey=runner.vkey, color=color_key)
    #     plt.savefig(f"{args.save_dir}/velocity_{basis}_time.svg", format='svg', dpi=300)
    #     plt.savefig(f"{args.save_dir}/velocity_{basis}_time.png", format='png', dpi=300)

if __name__ == "__main__":
    main()
