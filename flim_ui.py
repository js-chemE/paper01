import matplotlib.font_manager
import matplotlib.pyplot as plt

main_params = {
    # text
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 24,
    "pdf.fonttype": 42,
    # axes
    "axes.linewidth": 2,
    # "axes.edgecolor" : "grey",
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    # lines
    "lines.linewidth": 2,
    # scatter
    "lines.linewidth": 2,
    "lines.markersize": 30,
    # x-axis
    "xtick.top": True,
    "xtick.labelsize": 20,
    "xtick.major.size": 9,
    "xtick.minor.size": 6,
    "xtick.major.width": 2,
    "xtick.minor.width": 1.4,
    "xtick.major.pad": 6,
    "xtick.minor.pad": 6,
    "xtick.direction": "in",
    "xtick.minor.visible": True,
    # y-axis
    "ytick.right": True,
    "ytick.labelsize": 20,
    "ytick.major.size": 9,
    "ytick.minor.size": 6,
    "ytick.major.width": 2,
    "ytick.minor.width": 1.4,
    "ytick.major.pad": 6,
    "ytick.minor.pad": 6,
    "ytick.direction": "in",
    "ytick.minor.visible": True,
    # grid
    "axes.grid": True,
    "grid.color": "gray",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    # figure
    "figure.figsize": (8.25, 5),
    # savefig
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "savefig.transparent": True,
}


fullwidth_params = {
    # figure
    "figure.figsize": (17.8, 5),
}

PARAMS = {"main": main_params, "fullwidth": fullwidth_params}

plt.rcParams.update(main_params)


def update_params(params: dict) -> None:
    plt.rcParams.update(params)


def update_params_string(params: str, **kwargs) -> None:
    params_dict = PARAMS[params]
    for k, v in kwargs:
        params_dict[k] = v
    update_params(params_dict)
