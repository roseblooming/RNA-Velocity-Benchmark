import pandas as pd
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns

def extract(results, model, data_type, noise_type, metric):
    df = results[model][data_type][noise_type]
    df = df[metric].groupby(df['noise_level']).agg(['mean', 'std']).reset_index()
    df.set_index('noise_level', inplace=True)
    return df

def my_concate(grouped):
    mean_columns = [col for col in grouped.columns if col[1] == 'mean']
    comb_mean = pd.concat([grouped[col] for col in mean_columns], ignore_index=True)
    std_columns = [col for col in grouped.columns if col[1] == 'std']
    comb_std = pd.concat([grouped[col] for col in std_columns], ignore_index=True)
    cols = grouped.columns.get_level_values(0).unique()
    noise = grouped.index.tolist()
    data = {
        'model': cols.repeat(len(noise)),
        'noise': noise * len(cols),
        'mean': comb_mean,
        'std': comb_std
    }
    data = pd.DataFrame(data)
    return data

def plot_metric(data:pd.DataFrame, metric, save_as):
    # plt.figure(figsize=(data['model'].nunique()+1,4))
    fig_width = data['model'].nunique() + 1
    if metric == 'time_corr_spearman':
        y_low = -1
        fig_height = 6
    elif metric == 'velocity_acc':
        y_low = -0.4
        fig_height = 5
    else:
        y_low = 0
        fig_height = 4
    plt.figure(figsize=(fig_width, fig_height))
    palette=sns.blend_palette(['#FFE7BA', '#8B7E66'], as_cmap=True)
    sns.barplot(
        data=data,
        x='model',
        y='mean',
        hue='noise',
        ci=None,
        palette=palette
    )
    bars = plt.gca().patches
    for i in range(data['model'].nunique()):
        for j in range(data['noise'].nunique()):
            bar = bars[j * data['model'].nunique() + i]
            if bar.get_height() != 0:
                # 获取对应的误差
                error = data['std'].iloc[i * data['noise'].nunique() + j]
                # 获取柱子的 x 和 y 坐标
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                # 绘制误差条
                plt.errorbar(
                    x=x, 
                    y=y, 
                    yerr=error, 
                    fmt='none', 
                    ecolor='gray', 
                    capsize=2,
                    capthick=0.5,
                    elinewidth=0.5
                )
    plt.ylim(y_low, 1)
    plt.xlabel('model')
    plt.ylabel(metric)
    plt.legend(title='noise level', loc='upper right')
    plt.tight_layout()
    plt.savefig(save_as, dpi=300, format='svg')
    plt.show()