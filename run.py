import sys
sys.path.append("models")

import scanpy as sc
# from Runner.deepvelo_cui_runner import DeepVeloCuiRunner
from Runner.unitvelo_runner import UniTVeloRunner
from Runner.velovi_runner import veloVI_Runner
from Runner.velovae_runner import VeloVAE_Runner
from Runner.scvelo_runner import scVeloRunner
from Runner.topovelo_runner import TopoVeloRunner
from MetricEvaluator.evaluator import BaseEvaluator
import argparse
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata_path", type=str, required=True, help="path to h5ad")
    parser.add_argument("--save_dir", type=str, default=None, help="save dir")
    parser.add_argument("--device", type=int, default=0, help="device to train")
    parser.add_argument("--model", type=str, required=True, choices=["scvelo", "topovelo", "velovae", "velovi", "unitvelo", "deepvelo_cui"], help="model to evaluate")
    parser.add_argument("--real_data", action="store_true", default=False, help="wheather to run on real data")
    parser.add_argument("--label", type=str, default=None, help="label key")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    adata = sc.read_h5ad(args.adata_path)
    if args.save_dir is None:
        args.save_dir = time.strftime(f"logs/{args.model}/%Y%m%d-%H%M%S", time.localtime())
    os.makedirs(args.save_dir, exist_ok=True)
    pd.Series(vars(args)).to_csv(f"{args.save_dir}/config.csv")
    if args.model == "scvelo":
        runner = scVeloRunner("dynamical", adata, args.real_data, save_dir=args.save_dir)
    elif args.model == "topovelo":
        runner = TopoVeloRunner(adata, "X_coord", args.real_data, device=args.device, save_dir=args.save_dir)
    elif args.model == "velovae":
        runner = VeloVAE_Runner(adata, args.real_data, device=args.device, save_dir=args.save_dir)
    elif args.model == "velovi":
        runner = veloVI_Runner(adata, args.real_data, device=args.device, save_dir=args.save_dir)
    elif args.model == "unitvelo":
        runner = UniTVeloRunner(adata, args.real_data, args.device, args.save_dir, args.label)
    # elif args.model == "deepvelo_cui":
    #     runner = DeepVeloCuiRunner(adata, args.real_data, device=args.device, save_dir=args.save_dir)
    runner.run()
    if args.model == "scvelo" and "velocity_genes" in adata.var:
        adata = adata[:, adata.var["velocity_genes"]].copy()
    if args.model in ["unitvelo", "velovi"]:
        adata = runner.adata
    evaluater = BaseEvaluator()
    evaluater.evaluate(adata, runner, runner.vkey, runner.tkey, runner.u_hat_key, runner.s_hat_key)
    metric = evaluater.metric
    metric_series = metric.to_series()
    metric_series.to_csv(f"{args.save_dir}/metric.csv")
    if args.real_data == False:
        color_key = "true_t"
    else:
        color_key = "pred_t"
    if "X_coord" in adata.obsm.keys():
        evaluater.draw_spatial_velocity(adata, basis="coord", vkey=runner.vkey, color=color_key)
        plt.savefig(f"{args.save_dir}/velocity_spatial.svg", format='svg', dpi=600)
        plt.savefig(f"{args.save_dir}/velocity_spatial.png", format='png', dpi=600)
    sc.pp.pca(adata)
    evaluater.draw_spatial_velocity(adata, basis="pca", vkey=runner.vkey, color=color_key)
    plt.savefig(f"{args.save_dir}/velocity_pca.svg", format='svg', dpi=600)
    plt.savefig(f"{args.save_dir}/velocity_pca.png", format='png', dpi=600)
    sc.tl.umap(adata)
    evaluater.draw_spatial_velocity(adata, basis="umap", vkey=runner.vkey, color=color_key)
    plt.savefig(f"{args.save_dir}/velocity_umap.svg", format='svg', dpi=600)
    plt.savefig(f"{args.save_dir}/velocity_umap.png", format='png', dpi=600)

if __name__ == "__main__":
    main()
