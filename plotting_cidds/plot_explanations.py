
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

from run_cidds_xai import xai_to_categorical


def plot_hamming_heatmap(data, save_path=None):
    data_filtered = data.apply(lambda row: row > 0.25 * row.max(), axis=1)
    dists = pdist(data_filtered, metric='hamming')
    heatmap = squareform(dists)

    # plot
    font = {'family': 'arial',
            'size': 28}
    matplotlib.rc('font', **font)
    matplotlib.rc('text', usetex=True)
    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap)

    # axis labels
    ticks = [0, 20, 40, 60, 80]
    label_pos_x = [0.5, 0.5, 0.5, 0.5, 0]
    label_pos_y = [-0.5, -0.5, -0.5, -0.5, 0]

    plt.xticks(ticks=ticks, rotation=0)
    ax.set_xticklabels(['dos', 'port', 'ping', 'brute', ''])
    for i, label in enumerate(ax.xaxis.get_majorticklabels()):
        dx = label_pos_x[i]
        offset = matplotlib.transforms.ScaledTranslation(dx, 0, fig.dpi_scale_trans)
        label.set_transform(label.get_transform() + offset)

    plt.yticks(ticks=ticks, rotation=0)
    ax.set_yticklabels(['dos', 'port', 'ping', 'brute', ''])
    for i, label in enumerate(ax.yaxis.get_majorticklabels()):
        dy = label_pos_y[i]
        offset = matplotlib.transforms.ScaledTranslation(0, dy, fig.dpi_scale_trans)
        label.set_transform(label.get_transform() + offset)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.show()


if __name__ == '__main__':

    approach = 'AE'  # 'AE', 'IF', 'OCSVM'
    background = 'optimized'  # 'zeros', 'mean', 'kmeans', 'NN', 'optimized'
    save = True

    expl = pd.read_csv(f'../outputs/explanation/cidds/{approach}_shap_{background}.csv', header=0, index_col=0)
    if 'expected_value' in expl.columns:
        expl = expl.drop('expected_value', axis=1)
    if approach in ['IF', 'OCSVM']:
        expl = -1 * expl
    expl = xai_to_categorical(expl)

    plot_hamming_heatmap(data=expl, save_path=f'./figures/heatmap_{approach}_{background}.png' if save else None)
