import sys

from Runner.deepvelo_cai_runner import DeepVeloCaiRunner
# from Runner.unitvelo_runner import UniTVeloRunner
# from Runner.velovi_runner import veloVI_Runner
# from Runner.velovae_runner import VeloVAE_Runner
sys.path.append("models")
import scanpy as sc
from Runner.scvelo_runner import scVeloRunner
from Runner.topovelo_runner import TopoVeloRunner
from MetricEvaluator.evaluator import BaseEvaluator
import argparse
import os
import time
import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata_path", type=str, required=True, help="path to h5ad")
    parser.add_argument("--save_dir", type=str, default=None, help="save dir")
    parser.add_argument("--device", type=int, required=True, help="device to train")
    parser.add_argument("--model", type=str, required=True, choices=["scvelo", "topovelo", "velovae", "velovi", "unitvelo", "deepvelo_cai"], help="model to evaluate")
    parser.add_argument("--real_data", action="store_true", default=False, help="wheather to run on real data")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # adata = sc.read_h5ad("/HDD1/chensishuo/spvelo_data/simulate/bidirect/exp_noise/0.5/20240605_222510/data_simu.h5ad")
    adata = sc.read_h5ad(args.adata_path)
    # adata = adata[:2000].copy()
    if args.save_dir is None:
        args.save_dir = time.strftime(f"logs/{args.model}/%Y%m%d-%H%M%S", time.localtime())
    os.makedirs(args.save_dir, exist_ok=True)
    pd.Series(vars(args)).to_csv(f"{args.save_dir}/config.csv")
    if args.model == "scvelo":
        runner = scVeloRunner("dynamical", adata, args.real_data, save_dir=args.save_dir)
    elif args.model == "topovelo":
        runner = TopoVeloRunner(adata, "X_coord", args.real_data, device=args.device, save_dir=args.save_dir)
    # elif args.model == "velovae":
    #     runner = VeloVAE_Runner(adata, args.real_data, device=args.device, save_dir=args.save_dir)
    # elif args.model == "velovi":
    #     runner = veloVI_Runner(adata, args.real_data, device=args.device, save_dir=args.save_dir)
    # elif args.model == "unitvelo":
    #     runner = UniTVeloRunner(adata, args.real_data, args.device, args.save_dir)
    elif args.model == "deepvelo_cai":
        runner = DeepVeloCaiRunner(adata, args.real_data, device=args.device, save_dir=args.save_dir)
    runner.run()
    # print(adata, adata.var)
    if args.model == "scvelo" and "velocity_genes" in adata.var:
        adata = adata[:, adata.var["velocity_genes"]].copy()
    evaluater = BaseEvaluator()
    print(adata)
    print(runner.adata)
    print(runner.vkey, runner.tkey, runner.u_hat_key, runner.s_hat_key)
    if args.model == "unitvelo":
        evaluater.evaluate(runner.adata, runner, runner.vkey, runner.tkey, runner.u_hat_key, runner.s_hat_key)
    else:
        evaluater.evaluate(adata, runner, runner.vkey, runner.tkey, runner.u_hat_key, runner.s_hat_key)
    metric = evaluater.metric
    metric_series = metric.to_series()
    metric_series.to_csv(f"{args.save_dir}/metric.csv")
    print(evaluater.metric)
    evaluater.draw_spatial_velocity(adata, basis="coord", vkey=runner.vkey, color="true_t")
    plt.savefig(f"{args.save_dir}/velocity_spatial.png", dpi=600)
    sc.pp.pca(adata)
    evaluater.draw_spatial_velocity(adata, basis="pca", vkey=runner.vkey, color="true_t")
    plt.savefig(f"{args.save_dir}/velocity_pca.png", dpi=600)

if __name__ == "__main__":
    # args = parse_args()
    # main_process_pid = multiprocessing.current_process().pid
    # monitor_process = multiprocessing.Process(target=monitor_memory_usage, args=(main_process_pid,))
    # monitor_process_gpu = multiprocessing.Process(target=monitor_gpu_memory_usage, args=(main_process_pid, args.device))
    # monitor_process.start()
    # monitor_process_gpu.start()
    main()
    # main(args)
    # monitor_process.join()
    # monitor_process_gpu.join()
