import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from genexna.evaluate import segregate_2d


def _hard_network_avg(val_df, network_labels):
    """
    Calculate the average metric value for all the networks, against each trait.
    """
    columns = val_df.columns

    # Segregate the values for corresponding networks.
    network_pcc = segregate_2d(val_df.to_numpy(), network_labels)

    # Calculate the average value for each network.
    avg_val = np.vstack([network_pcc[network_i].mean(axis=0)
                        for network_i in set(network_labels)])

    # Convert the numpy array into a data-frame.
    return pd.DataFrame(avg_val, index=set(network_labels), columns=columns)


def _soft_network_avg(val_df, gene_network_prob):
    """
    Calculate the average metric value for all the soft networks, against each trait.
    """
    avg_val = np.zeros((gene_network_prob.shape[0], val_df.shape[1]))

    # Calculate the average value for each network.
    total_network_probs = gene_network_prob.sum(axis=1)
    for gene in gene_network_prob:
        for network, prob in gene_network_prob[gene].items():
            avg_val[network, :] = np.add(
                avg_val[network, :],
                np.nan_to_num(val_df.loc[gene, :].to_numpy() * prob / total_network_probs[network]))
    return pd.DataFrame(data=avg_val, index=range(gene_network_prob.shape[0]), columns=val_df.columns)


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Args:
        data: A 2D numpy array of shape (N, M)
        row_labels: A list of length N with the labels for the rows
        col_labels: A list of length M with the labels for the columns
        ax: A `matplotlib.axes.Axes` instance to which the heatmap
      is plotted. If not provided, use current axes or create a new one.
        cbar_kw: A dictionary with arguments to `matplotlib.Figure.colorbar`.
        cbarlabel: The label for the colorbar.
    """
    if cbar_kw is None:
        cbar_kw = {}
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, val_fmt='{x:.2f}', text_colors=None, threshold=None, **kwargs):
    """
    A function to annotate a heatmap.

    Args:
        im: The AxesImage to be labeled.
        data: Data used to annotate. If None, the image's data is used.
        val_fmt: The format of the annotations inside the heatmap.
      This should either use the string format method, e.g. "$ {x:.2f}",
      or be a `matplotlib.ticker.Formatter`.
        text_colors: A list or array of two color specifications.
      The first is used for values below a threshold, the second for those
      above.
        threshold: Value in data units according to which the colors from
      text_colors are applied. If None (the default) uses the middle of the
      colormap as separation.
    """
    text_colors = ['black', 'white'] if text_colors is None else text_colors
    data = im.get_array() if not isinstance(data, (list, np.ndarray)) else data

    # Normalize the threshold to the images color range.
    threshold = im.norm(threshold) if threshold else im.norm(np.max(data)) / 2

    # Set default alignment to center, but allow it to be overwritten by kwargs.
    kw = {'horizontalalignment': 'center', 'verticalalignment': 'center'}
    kw.update(kwargs)

    # Get the formatter in case a string is supplied.
    val_fmt = matplotlib.ticker.StrMethodFormatter(val_fmt) if isinstance(val_fmt, str) else val_fmt

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=text_colors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, val_fmt(data[i, j], None), **kw)
            texts.append(text)
    return texts


def module_trait_heat_map(
    pccs, p_values, network_labels, corr_file_path, p_value_file_path, 
    soft=False, max_heatmap_rows=5
):
    """
    Plot a heat map for module-trait correlations.
    """
    pcc_corr_map = _soft_network_avg(
        pccs, network_labels) if soft else _hard_network_avg(pccs, network_labels)
    p_value_corr_map = _soft_network_avg(
        p_values, network_labels) if soft else _hard_network_avg(p_values, network_labels)
    if pcc_corr_map.shape[0] > max_heatmap_rows:
        pcc_corr_map['avg'] = p_value_corr_map['avg'] = (pcc_corr_map.abs() * (-pcc_corr_map + 0.05)).mean(
            numeric_only=True, axis=1)
        pcc_corr_map = pcc_corr_map.nlargest(
            max_heatmap_rows, 'avg').drop(columns=['avg'])
        p_value_corr_map = p_value_corr_map.nlargest(
            max_heatmap_rows, 'avg').drop(columns=['avg'])
    row_labels = [f'network #{i}' for i in range(pcc_corr_map.shape[0])]

    # Plot the correlation matrices as a heat-map.
    im, _ = heatmap(data=pcc_corr_map.to_numpy(dtype=float),
                       row_labels=row_labels,
                       col_labels=pcc_corr_map.columns,
                       ax=plt.axes(),
                       cmap='YlGn',
                       cbarlabel='Pearson Correlation')
    annotate_heatmap(im, val_fmt='{x:.2f}')
    plt.savefig(corr_file_path)
    plt.clf()

    im, _ = heatmap(data=p_value_corr_map.to_numpy(dtype=float),
                       row_labels=row_labels,
                       col_labels=p_value_corr_map.columns,
                       ax=plt.axes(),
                       cmap='PuRd',
                       cbarlabel='Pearson P Value')
    annotate_heatmap(im, val_fmt='{x:.2g}')
    plt.savefig(p_value_file_path)
    plt.clf()
