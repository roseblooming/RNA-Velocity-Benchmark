import sys
sys.path.append("models")

import os
import argparse
import pandas as pd
import scanpy as sc
from Runner.BaseRunner import BaseRunner
from MetricEvaluator.evaluator import BaseEvaluator
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata_path", type=str, required=True, help="path to h5ad")
    parser.add_argument("--save_dir", type=str, default=None, help="save dir")
    parser.add_argument("--device", type=int, default=None, help="device to train")
    parser.add_argument("--model", type=str, required=True, choices=["scvelo", "topovelo", "velovae", "velovi", "unitvelo", "deepvelo_cui", "stt", "latentvelo", "velovgi"], help="model to evaluate")
    parser.add_argument("--label", type=str, default=None, help="label or cluster key")
    parser.add_argument("--basis", type=str, nargs='+', required=True, help="basis to evaluate and plot")
    parser.add_argument("--spatial_key", type=str, default=None, help="spatial key")
    parser.add_argument("--batch_key", type=str, default=None, help="batch key")
    parser.add_argument("--cluster_edges_path", type=str, default=None, help="path to cluster edges")

    parser.add_argument("--tkey_true", type=str, default=None, help="true time key, true_t for simu data")
    parser.add_argument("--vkey_true", type=str, default=None, help="true velocity key, true_velocity for simu data")
    parser.add_argument("--is_data_simu", action="store_true", default=False, help="whether the data is simulated using method proposed in topovelo")

    parser.add_argument("--redo", action="store_true", default=False, help="use result from previous run")
    parser.add_argument("--do_not_plot", action="store_true", default=False, help="do not plot")
    parser.add_argument("--do_not_eval", action="store_true", default=False, help="do not evaluate")

    parser.add_argument("--batch_correction", action="store_true", default=False, help="whether to do batch correction")
    parser.add_argument("--n_branches", type=int, default=None, help="number of branches (only for latentvelo)")
    parser.add_argument("--n_states", type=int, default=None, help="number of states (only for stt)")
    parser.add_argument("--use_annotation", action="store_true", default=False, help="whether to use annotation (only for latentvelo)")
    parser.add_argument("--use_tprior", action="store_true", default=False, help="whether to use tprior (for latentvelo and velovae)")
    parser.add_argument("--sample", action="store_true", default=False, help="whether to sample (only for velovgi)")
    parser.add_argument("--spatial_graph_method", type=str, default=None, help="spatial graph method (only for topovelo)")
    parser.add_argument("--adata2_path", type=str, default=None, help="path to aggregated/latent adata (for stt and latentvelo when redo is True)")

    parser.add_argument("--dimred_path", type=str, default=None, help="path to dimred csv")
    parser.add_argument("--varn_unique", action="store_true", default=False, help="make var names unique (required for hsc)")
    parser.add_argument("--del_ori_nbs", action="store_true", default=False, help="delete original neighbors")
    parser.add_argument("--pp_choice", type=int, required=True, choices=[0, 1, 2, 3], help="0: default, 1: only moments, 2: no preprocess, 3: no filter")
    parser.add_argument("--n_hvg", type=int, default=2000, help="number of highly variable genes")
    parser.add_argument("--leiden_resolution", type=float, default=None, help="resolution for leiden clustering (only for simu data, 0.01 for bidirectional, 0.1 for radial)")
    
    # parser.add_argument("--use_rep_for_nbs", action="store_true", default=False, help="whether compute neighbors using specific basis")
    # parser.add_argument("--basis_key_add_prefix", type=str, nargs='+', default=None, help="add prefix to basis key")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    adata = sc.read_h5ad(args.adata_path)
    if args.redo and args.adata2_path is not None:
        adata2 = sc.read_h5ad(args.adata2_path)
    # if args.basis_key_add_prefix is not None:
    #     for basis in args.basis_key_add_prefix:
    #         adata.obsm[f'X_{basis}'] = adata.obsm[basis]
    if args.varn_unique:
        adata.var_names_make_unique()
    if args.dimred_path is not None:
        apply_dimred_to_adata(adata, args.dimred_path)
    if args.del_ori_nbs and 'neighbors' in adata.uns.keys():
        del adata.uns['neighbors']
    if args.leiden_resolution is not None:
        perform_leiden_clustering(adata, args.leiden_resolution)    
    if args.cluster_edges_path is not None:
        import pickle
        with open(args.cluster_edges_path, "rb") as f:
            cluster_edges = pickle.load(f)    
    # adata.obsm["X_pca_backup"] = adata.obsm["X_pca"].copy()
    # adata.obsm["X_umap_backup"] = adata.obsm["X_umap"].copy()
    if args.save_dir is None:
        import time
        args.save_dir = time.strftime(f"logs/{args.model}/%Y%m%d-%H%M%S", time.localtime())
    
    os.makedirs(args.save_dir, exist_ok=True)
    pd.Series(vars(args)).to_csv(path_or_buf=f"{args.save_dir}/config.csv" if args.redo == False else f"{args.save_dir}/config_redo.csv")
    
    if args.model == "scvelo":
        from Runner.scvelo_runner import scVeloRunner
        runner = scVeloRunner(
            model="dynamical",
            adata=adata,
            # is_real=args.real_data,
            pp_choice=args.pp_choice,
            n_hvg=args.n_hvg,
            save_dir=args.save_dir,
            label=args.label
        )
    elif args.model == "topovelo":
        from Runner.topovelo_runner import TopoVeloRunner
        runner = TopoVeloRunner(
            adata=adata,
            spatial_key=args.spatial_key,
            spatial_graph_method=args.spatial_graph_method,
            # is_real=args.real_data,
            is_data_simu=args.is_data_simu,
            pp_choice=args.pp_choice,
            n_hvg=args.n_hvg,
            device=args.device,
            save_dir=args.save_dir,
            label=args.label,
            t_prior=args.use_tprior
        )
    elif args.model == "velovae":
        from Runner.velovae_runner import VeloVAE_Runner
        runner = VeloVAE_Runner(
            adata=adata,
            # is_real=args.real_data,
            is_data_simu=args.is_data_simu,
            pp_choice=args.pp_choice,
            n_hvg=args.n_hvg,
            device=args.device,
            save_dir=args.save_dir,
            label=args.label,
            t_prior=args.use_tprior
        )
    elif args.model == "velovi":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        from Runner.velovi_runner import veloVI_Runner
        runner = veloVI_Runner(
            adata=adata,
            # is_real=args.real_data,
            pp_choice=args.pp_choice,
            n_hvg=args.n_hvg,
            device=args.device,
            save_dir=args.save_dir,
            label=args.label
        )
    elif args.model == "unitvelo":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        from Runner.unitvelo_runner import UniTVeloRunner
        runner = UniTVeloRunner(
            adata=adata,
            # is_real=args.real_data,
            pp_choice=args.pp_choice,
            n_hvg=args.n_hvg,
            device=args.device,
            save_dir=args.save_dir,
            label=args.label
        )
    elif args.model == "deepvelo_cui":
        if not args.redo:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
            from Runner.deepvelo_cui_runner import DeepVeloCuiRunner
            runner = DeepVeloCuiRunner(
                adata=adata,
                # is_real=args.real_data,
                pp_choice=args.pp_choice,
                n_hvg=args.n_hvg,
                device=args.device,
                save_dir=args.save_dir,
                label=args.label
            )
        else: # so that no need to change environment
            runner = BaseRunner(
                model_name=args.model,
                adata=adata,
                # is_real=args.real_data,
                pp_choice=args.pp_choice,
                n_hvg=args.n_hvg,
                save_dir=args.save_dir,
                label=args.label
            )
    elif args.model == "stt":
        from Runner.stt_runner import STT_Runner
        runner = STT_Runner(
            adata=adata,
            # is_real=args.real_data,
            pp_choice=args.pp_choice,
            n_hvg=args.n_hvg,
            label=args.label,
            spatial_key=args.spatial_key,
            n_states=args.n_states,
            adata_aggr=adata2 if args.redo else None,
            save_dir=args.save_dir
        )
    elif args.model == "latentvelo":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        from Runner.latentvelo_runner import LatentVeloRunner
        runner = LatentVeloRunner(
            adata=adata,
            # is_real=args.real_data,
            batch_correction=args.batch_correction,
            device=args.device,
            zr_dim=min(args.n_branches-1, 1),
            label=args.label,
            batch_key=args.batch_key,
            latent_adata=adata2 if args.redo else None,
            pp_choice=args.pp_choice,
            n_hvg=args.n_hvg,
            compute_umap=False,
            save_dir=args.save_dir,
            annotated=args.use_annotation,
            t_prior=args.use_tprior,
            is_data_simu=args.is_data_simu
        )
    elif args.model == "velovgi":
        from Runner.velovgi_runner import VeloVGI_Runner
        runner = VeloVGI_Runner(
            adata=adata,
            # is_real=args.real_data,
            batch_correction=args.batch_correction,
            sample=args.sample,
            device=args.device,
            label=args.label,
            batch_key=args.batch_key,
            pp_choice=args.pp_choice,
            n_hvg=args.n_hvg,
            save_dir=args.save_dir
        )
    
    if args.redo == False:
        runner.run()
        # if args.model == "velovi" and args.real_data: # TODO: solve reference problem
        # adata = runner.adata # FIXME: be cautious of issues caused by this operation
        # TODO: 把adata的地方全部换成runner.adata
    # elif args.model == "stt":
    #     runner.adata_aggr = sc.read_h5ad(args.adata_aggr_path)

    # if args.model == "scvelo" and "velocity_genes" in adata.var:
    #     adata = adata[:, adata.var["velocity_genes"]].copy()
    # if args.model == 'topovelo':
    #     adata = adata[(adata.layers[runner.vkey] != 0).any(axis=1)].copy()
    #     del adata.uns['neighbors']
    #     del adata.obsp[runner.spatial_graph_key]

    # if args.model != "stt":
    #     from sklearn.preprocessing import MinMaxScaler
    #     scaler = MinMaxScaler()
    #     adata.obs[f'{runner.tkey}_norm'] = scaler.fit_transform(adata.obs[[runner.tkey]])
    # if args.model == "scvelo":
    #     adata.obs[f'{runner.t_gene_key}_mean'] = adata.layers[runner.t_gene_key].mean(axis=1)
    #     adata.obs[f'{runner.t_gene_key}_norm'] = scaler.fit_transform(adata.obs[[f'{runner.t_gene_key}_mean']])
    
    evaluator = BaseEvaluator(cluster_edges=cluster_edges if args.cluster_edges_path is not None else None, cluster_key=args.label, t_key_true=args.tkey_true, v_key_true=args.vkey_true, spatial_key=args.spatial_key)
    # for basis in args.basis:
        # if basis == 'umap' and 'X_umap' not in adata.obsm.keys():
        #     sc.tl.umap(adata)
    if not args.do_not_eval:
        evaluator.evaluate(runner, args.basis)
        metric = evaluator.metric
        # print(metric)
        metric_series = metric.to_series()
        metric_series.to_csv(path_or_buf=f"{args.save_dir}/metric.csv" if args.redo == False else f"{args.save_dir}/metric_redo.csv")
        
        # 使用字典和循环替换重复逻辑
        metrics_to_save = {
            "cbdir": metric.cbdir,
            "inner_cluster_coh": metric.inner_cluster_coh,
            "velocity_consistency": metric.velocity_consistency,
            "spatial_velo_consist": metric.spatial_velo_consist
        }
        save_metric_dict(metrics_to_save, args.save_dir)

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
        evaluator.plot_velocity_stream(runner, args.basis, args.save_dir)
        evaluator.plot_time(runner, args.basis, args.save_dir)
        # for basis in args.basis:
        # # plot(args.label, args.save_dir, adata, runner, evaluator, color_key, basis)
        #     evaluator.plot_velocity_stream(adata, basis, runner.vkey, args.save_dir, runner.adata_aggr if args.model == "stt" else None)
        #     # plt.savefig(f"{args.save_dir}/velocity_{basis}.svg", format='svg', dpi=300)
        #     if args.model != "stt":
        #         evaluator.plot_time(adata, basis, f'{runner.tkey}_norm' if not args.do_not_eval else runner.tkey, args.save_dir, f'{runner.t_gene_key}_norm' if args.model == "scvelo" else None)
        #         # plt.savefig(f"{args.save_dir}/time_{basis}.svg", format='svg', dpi=300)

        # evaluator.compute_and_plot_PAGA_graph(adata, runner.vkey, f'{runner.tkey}_norm', basis)
        # plt.savefig(f"{args.save_dir}/PAGA_{basis}.svg", format='svg', dpi=300)
        # evaluator.transition_confidence.to_csv(f"{args.save_dir}/transition_confidence_{basis}.csv")
        ###### 保留 ######

    # sc.pp.pca(adata)
    # adata.obsm["X_pca"] = adata.obsm["X_pca_backup"].copy()
    # adata.obsm["X_umap"] = adata.obsm["X_umap_ori"].copy()
    # plot(args.label, args.save_dir, adata, runner, evaluator, color_key, "pca")

def save_metric_dict(metrics_to_save, save_dir):
    """
    保存所有 metrics 到指定目录。

    :param metrics_to_save: 包含所有 metrics 的字典
    :param save_dir: 保存目录
    """
    import pickle
    for prefix, metric_dict in metrics_to_save.items():
        if isinstance(metric_dict, dict):
            for key, value in metric_dict.items():
                with open(f"{save_dir}/{prefix}_{key}.pkl", "wb") as f:
                    pickle.dump(value, f)

def apply_dimred_to_adata(adata, dimred_path):
    """
    将降维结果应用到 AnnData 对象。

    :param adata: AnnData 对象
    :param dimred_path: 降维结果的 CSV 文件路径
    """
    dimred_df = pd.read_csv(dimred_path, index_col=0)
    dimred_df.index = dimred_df.index.astype(str)
    dimred_df = dimred_df.loc[adata.obs.index]
    for basis in ['umap', 'tsne', 'phate']:
        if f'X_{basis}_1' in dimred_df.columns:
            adata.obsm[f'X_{basis}'] = dimred_df[[f'X_{basis}_1', f'X_{basis}_2']].values

def perform_leiden_clustering(adata, resolution):
    """
    对 AnnData 对象执行 Leiden 聚类，。

    :param adata: AnnData 对象
    :param resolution: Leiden 聚类的分辨率参数
    """
    adata_orig = adata.copy()
    adata_orig.X = adata_orig.layers['true_spliced'].copy()
    adata_orig.layers['spliced'] = adata_orig.layers['true_spliced'].copy()
    adata_orig.layers['unspliced'] = adata_orig.layers['true_unspliced'].copy()
    sc.tl.pca(adata_orig, n_comps=50)
    sc.pp.neighbors(adata_orig, n_neighbors=50, n_pcs=50)
    sc.tl.leiden(adata_orig, resolution=resolution)
    adata.obs['leiden'] = adata_orig.obs['leiden']
    adata.obsm['X_pca_orig'] = adata_orig.obsm['X_pca']

# NOTE: not used
# def plot(label, save_dir, adata, runner:BaseRunner, evaluator:BaseEvaluator, color_key, basis):
#     if label is not None:
#         evaluator.plot_velocity_stream(adata, basis=basis, vkey=runner.vkey, color=label)
#         plt.savefig(f"{save_dir}/velocity_{basis}_label.svg", format='svg', dpi=300)
#         if color_key is not None:
#             scv.pl.scatter(adata, basis=basis, color=color_key, color_map="viridis")
#             plt.savefig(f"{save_dir}/{basis}_time.svg", format='svg', dpi=300)
#     elif color_key is not None:
#         evaluator.plot_velocity_stream(adata, basis=basis, vkey=runner.vkey, color=color_key, color_map="viridis")
#         plt.savefig(f"{save_dir}/velocity_{basis}_time.svg", format='svg', dpi=300)

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
    # session_name = os.getenv("TMUX_SESSION_NAME")  # 从环境变量获取会话名称
    # if session_name: # 如果在 tmux 会话中运行，发送任务完成信号
    #     os.system(f"tmux wait-for -S finished-{session_name}")
