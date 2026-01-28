import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
import anndata as ad

def find_lambda(adata:ad.AnnData, target_dropout_rate, upper_bound):
    """
    计算能使联合dropout率达到目标值的lambda参数
    
    参数:
    adata: AnnData对象
    target_dropout_rate: 目标联合dropout率 (0~1之间)
    upper_bound: lambda的上界
    
    返回:
    lambda值
    """
    # 从AnnData对象中提取数据
    unspliced_matrix = adata.layers['unspliced']
    spliced_matrix = adata.layers['spliced']

    # 转换为COO格式以便于操作非零元素
    if not sp.isspmatrix_coo(unspliced_matrix):
        u_matrix_coo = unspliced_matrix.tocoo()
    else:
        u_matrix_coo = unspliced_matrix.copy()
    
    if not sp.isspmatrix_coo(spliced_matrix):
        s_matrix_coo = spliced_matrix.tocoo()
    else:
        s_matrix_coo = spliced_matrix.copy()

    # 获取非零元素的值
    u_data = u_matrix_coo.data
    s_data = s_matrix_coo.data

    # 合并两个矩阵的数据以计算联合dropout率
    combined_data = np.concatenate([u_data, s_data])
    
    # 定义目标函数：理论dropout率与目标dropout率之间的差的平方
    def objective(lambda_val):
        if lambda_val < 0:
            return float('inf')  # lambda不能为负
        
        dropout_probs = np.exp(-lambda_val * np.abs(combined_data))
        theoretical_rate = np.mean(dropout_probs)
        return (theoretical_rate - target_dropout_rate) ** 2
    
    # 使用优化方法找到最合适的lambda值
    result = minimize_scalar(objective, bounds=(0, upper_bound), method='bounded')
    
    if not result.success:
        raise ValueError("无法找到合适的lambda值")
    
    return result.x

def dropout(adata:ad.AnnData, lambda_val, seed=None):
    """
    对RNA velocity的unspliced和spliced矩阵执行dropout, 保持相同的lambda值
    
    参数:
    adata: AnnData对象
    lambda_val: lambda值
    seed: 随机数种子, 用于可重复性 (可选)
    
    返回:
    dropout后的AnnData对象, 联合dropout率, (unspliced dropout率, spliced dropout率)
    """
    if seed is not None:
        np.random.seed(seed)

    # 从AnnData对象中提取数据
    unspliced_matrix = adata.layers['unspliced']
    spliced_matrix = adata.layers['spliced']

    # 转换为COO格式以便于操作非零元素
    if not sp.isspmatrix_coo(unspliced_matrix):
        u_matrix_coo = unspliced_matrix.tocoo()
    else:
        u_matrix_coo = unspliced_matrix.copy()
    
    if not sp.isspmatrix_coo(spliced_matrix):
        s_matrix_coo = spliced_matrix.tocoo()
    else:
        s_matrix_coo = spliced_matrix.copy()
    
    # 获取非零元素的值和坐标
    u_data = u_matrix_coo.data
    u_rows = u_matrix_coo.row
    u_cols = u_matrix_coo.col
    
    s_data = s_matrix_coo.data
    s_rows = s_matrix_coo.row
    s_cols = s_matrix_coo.col
    
    # 计算dropout概率
    u_dropout_probs = np.exp(-lambda_val * np.abs(u_data))
    s_dropout_probs = np.exp(-lambda_val * np.abs(s_data))
    
    # 执行随机dropout
    u_random_values = np.random.random(len(u_data))
    u_mask = u_random_values >= u_dropout_probs  # 保留元素的掩码
    
    s_random_values = np.random.random(len(s_data))
    s_mask = s_random_values >= s_dropout_probs  # 保留元素的掩码
    
    # 应用掩码，仅保留未被dropout的元素
    new_u_data = u_data[u_mask]
    new_u_rows = u_rows[u_mask]
    new_u_cols = u_cols[u_mask]
    
    new_s_data = s_data[s_mask]
    new_s_rows = s_rows[s_mask]
    new_s_cols = s_cols[s_mask]
    
    # 创建新的稀疏矩阵
    new_u_matrix = sp.coo_matrix((new_u_data, (new_u_rows, new_u_cols)), shape=unspliced_matrix.shape)
    new_s_matrix = sp.coo_matrix((new_s_data, (new_s_rows, new_s_cols)), shape=spliced_matrix.shape)
    
    # 计算实际dropout率
    u_actual_rate = 1.0 - (len(new_u_data) / len(u_data))
    s_actual_rate = 1.0 - (len(new_s_data) / len(s_data))
    
    # 计算实际联合dropout率
    total_elements = len(u_data) + len(s_data)
    kept_elements = len(new_u_data) + len(new_s_data)
    combined_actual_rate = 1.0 - (kept_elements / total_elements)
    
    # 将矩阵转换回原始格式
    new_u_matrix = new_u_matrix.asformat(unspliced_matrix.format)
    new_s_matrix = new_s_matrix.asformat(spliced_matrix.format)

    adata_out = adata.copy()
    adata_out.X = new_s_matrix
    adata_out.layers['unspliced'] = new_u_matrix
    adata_out.layers['spliced'] = new_s_matrix
    
    # 返回结果
    return adata_out, combined_actual_rate, (u_actual_rate, s_actual_rate)
