RNA Velocity Benchmark Repository

# Environments
- velo_env_new: Except for DeepVelo (Cui et al) and STT
- velo_env_stt: Except for UniTVelo, DeepVelo (Cui et al) and veloVI
- deepvelo_cui: DeepVelo (Cui et al), scVelo and TopoVelo; Not capable of plotting

# Directories
- models: Method src code for TopoVelo and STT
- Runner: Caller of each method for processing pipeline
- MetricEvaluator: Evaluation Module
- utils: Tools for memory monitoring and plotting
- run.py: Entrance file
- plots: Notebooks for metric plotting
- clusters: Ground truth cell types
- cluster_edges: Ground truth differentiation relationships between cell types
- env_config: Environment configuration
