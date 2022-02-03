import matplotlib.pyplot as plt
import pandas as pd


def __set_idx_safe(d: pd.DataFrame, idx_col):
    """
    Decides what to plot against based on given parameter
    """
    return d.index if idx_col is None or idx_col not in d.columns else d[idx_col]


def plot_all_together(d: pd.DataFrame, idx_col=None, normalize_col=None, normalize_by_last=False, log_scale=False,
                      exclude_cols=None, figsize=(15, 9)):
    """
    Plots all columns in a dataframe against an index column together
    """
    assert (idx_col is None or idx_col in d.columns) and (normalize_col is None or normalize_col in d.columns), \
        'Columns passed as params have to be present in the dataframe'

    if exclude_cols is None:
        exclude_cols = []

    idx = __set_idx_safe(d, idx_col)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    # add lines one by one to the same graph
    for col in d.columns:
        if col in exclude_cols:
            continue
        # ignore index column
        if idx_col is None or col == idx_col:
            continue

        # normalize
        if normalize_col is None:
            to_plot = d[col]
        else:
            d_ref, d_curr = d[normalize_col], d[col]
            factor = d_ref.iloc[-1] / d_curr.iloc[-1] if normalize_by_last else d_ref.iloc[1] / d_curr.iloc[1]
            to_plot = d_curr.apply(lambda x: x * factor)
        # ax.plot(idx, to_plot, color=__random_color_hex(), label=col)
        ax.plot(idx, to_plot, label=col)

    if log_scale:
        ax.set_yscale('log')
    plt.legend(loc="upper left")
    plt.show()


def plot_all_subplots(d: pd.DataFrame, idx_col=None, n_cols=2, figsize=None, exclude_cols=None):
    """
    Plots all columns in a dataframe against an index column, each into separate subgraph
    """
    if exclude_cols is None:
        exclude_cols = []
    assert idx_col is None or idx_col in d.columns, 'Index column has to be present when it is defined!'
    assert all([x in d.columns for x in exclude_cols]), 'All excluded columns must be present in the dataframe'

    if n_cols <= 0:
        return

    # count layout parameters
    n_subplots = len(d.columns) if idx_col is None else len(d.columns) - 1
    n_subplots -= len(exclude_cols)
    n_rows = ((n_subplots - 1) // n_cols) + 1

    # define index
    idx = __set_idx_safe(d, idx_col)
    # set figure size
    if figsize is None:
        fig_w = 10 if n_cols <= 1 else 20
        fig_h = 6 * n_rows
        figsize = (fig_w, fig_h)
    plt.figure(figsize=figsize)

    n_skipped = 0
    for i, col in enumerate(d.columns):
        if col in exclude_cols:
            n_skipped += 1
            continue
        # ignore index column
        if idx_col is None or col == idx_col:
            n_skipped += 1
            continue
        # plot new subplot
        plt.subplot(n_rows, n_cols, i - n_skipped + 1)
        plt.plot(idx, d[col])
        plt.title(col)

    plt.show()

