import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plot_discrepancies(
    y: np.ndarray,
    y_hat: np.ndarray,
    x_label: str = "Samples",
    y_label: str = "Objective function",
    title: str = "Sorted objective function values",
):

    I = np.argsort(y)

    x = np.arange(len(y))
    ax = plt.subplots()[1]
    ax.plot(
        x, y[I], c="tab:blue", ls="-", marker="o", label="Numerical model", markersize=3
    )
    ax.plot(
        x,
        y_hat[I],
        c="tab:orange",
        ls="-",
        marker="s",
        label="Surrogate model",
        markersize=3,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    # ax.set_xticks(x)
    plt.legend()
    plt.show()

    return ax


def variable_screeing_bar_plot(
    F_mean: np.ndarray,
    F_std: np.ndarray,
    var: list,
    save_fig: bool = False,
    path: str = "variable_screening_bar.pdf",
):

    """Plots the mean and std of the input variables previously computed.

    Arguments:
        F_mean {np.ndarray} -- mean values of elem. effect matrix F
        F_std {np.ndarray} -- std values of elem. effect matrix F
        var {list} -- list of variable names
        save_fig {bool} -- flag to save figure or not
    Returns:
        None
    """

    N = len(F_mean)

    score = F_mean ** 2 + F_std ** 2
    I = np.flipud(np.argsort(score))

    F_mean = np.abs(F_mean[I])
    F_std = np.abs(F_std[I])

    ind = np.arange(N)  # the x locations for the groups
    width = 0.65  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, F_mean / F_mean.mean(), width)
    p2 = plt.bar(ind, F_std / F_std.mean(), width, bottom=F_mean / F_mean.mean())

    plt.ylabel("Scaled mean and std by their means")
    plt.xlabel("Variables")
    plt.title("Variable screening results")
    plt.xticks(ind, var)
    plt.legend((p1[0], p2[0]), ("Mean", "Std"))

    plt.show()

    if save_fig:
        plt.savefig(
            path,
            dpi=300,
            format="pdf",
        )


def variable_distplot(
    X, y, var_names, a, b, n_bins, path="var_distplot.pdf", save_fig=False
):

    k = X.shape[1]

    I = np.argsort(y)

    X = X[I, :]
    y = y[I]

    in_space = (y > a) & (y < b)

    X_in = X[in_space, :]

    axs = plt.subplots(k, 1)[1]
    axs[0].set_title(f"Objective range [{a}, {b}]")
    for ix, ax in enumerate(axs):
        sns.distplot(
            X_in[:, ix],
            bins=np.linspace(0, 1, n_bins),
            ax=ax,
            norm_hist=True,
            kde_kws={"clip": [0, 1], "lw": 3.5, "ls": "--"},
        )
        ax.set_xlim([0, 1])
        ax.set_ylabel(var_names[ix])

    plt.tight_layout()
    plt.show()

    if save_fig:
        plt.savefig(
            path,
            dpi=300,
            format="pdf",
        )


def variable_screeing_scatter_plot(
    F_mean: np.ndarray,
    F_std: np.ndarray,
    var: list,
    save_fig: bool = False,
    path: str = "variable_screening_scatter.pdf",
):

    """Plots the mean and std of the input variables previously computed.

    Arguments:
        F_mean {np.ndarray} -- mean values of elem. effect matrix F
        F_std {np.ndarray} -- std values of elem. effect matrix F
        var {list} -- list of variable names
        save_fig {bool} -- flag to save figure or not
    Returns:
        None
    """

    ax = plt.subplots()[1]
    for i in range(len(var)):
        ax.text(F_mean[i], F_std[i], var[i])

    ax.set_xlim([np.min(F_mean), np.max(F_mean)])
    ax.set_ylim([np.min(F_std), np.max(F_std)])
    ax.grid()
    ax.set_xlabel("Mean")
    ax.set_ylabel("Standard deviation")
    if save_fig:
        plt.savefig(
            path,
            dpi=300,
            format="pdf",
        )

    return ax


def sorted_variables_bar_plot(
    ax: object,
    F_mean: np.ndarray,
    F_std: np.ndarray,
    var_names: list,
    titl: str,
    verbose=False,
):
    """[summary]

    Args:
        ax (object): matplotlib axis object
        F_mean (np.ndarray): mean values of elem. effect matrix F
        F_std (np.ndarray): std values of elem. effect matrix F
        var_names (list): list of variable names
        titl (str): title string
        verbose (bool, optional): Print variable ordering and values. Defaults to False.
    """

    dists = np.sqrt(F_mean ** 2 + F_std ** 2)
    sort_ind = np.argsort(dists)

    sort_ind = np.flipud(sort_ind)
    names_sorted = [var_names[ind] for ind in sort_ind]

    if verbose:
        print(names_sorted)
        # print(dists[sort_ind])

    x = np.arange(len(var_names))

    # fig, ax = plt.subplots(figsize=(7,3))

    ax.bar(x, dists[sort_ind])
    ax.set_xticks(x)
    ax.set_xticklabels(names_sorted)
    # ax.set_xlabel("Variables")
    ax.set_ylabel("General influence")
    ax.set_title(titl)
