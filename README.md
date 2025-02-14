### 环境
- velo_env_new: 除了deepvelo_cui和stt
- velo_env_stt: 除了unitvelo, deepvelo_cui & velovi
- deepvelo_cui: deepvelo_cui, scvelo & topovelo && 不能画图
- scgen: 用于去批次

### 项目结构
- models: 模型源代码 topovelo & stt
- runner: 调用每个模型执行流水线操作
- metric_evaluator: 评估模块
- utils: 工具模块 内存监控 && 绘图
- run.py: 最终执行的文件

- plots: 用于绘制指标图
- scripts: 用于模拟批次和去批次

- cluster_edges && clusters: 类型分化次序的边和细胞类型
- env_config: 环境配置
- result: 结果图 可视化和指标