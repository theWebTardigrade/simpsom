from typing import Optional, Union, Collection, Tuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylettes import Distinct20

from simpsom.polygons import Polygon

########################################################
# Edited by M. Pólvora Fonseca 02/05/2025

def plot_map(centers: Collection[np.ndarray], feature: Collection[np.ndarray], polygons_class: Polygon,
             show: bool = True, print_out: bool = False,
             file_name: str = "./som_plot.png", ax: Optional[Union[plt.Axes, Sequence[plt.Axes]]] = None,
             **kwargs: Tuple[int]) -> Tuple[Figure, plt.Axes]:
    """A simple line plot with maplotlib.

    Args: 
        y_val (array or list): values along the y axis.
        x_val (array or list): values along the x axis,
            if none, these will be inferred from the shape of y_val.
        show (bool): Choose to display the plot.
        print_out (bool): Choose to save the plot to a file.
        file_name (str): Name of the file where the plot will be saved if
            print_out is active. Must include the output path.
        kwargs (dict): Keyword arguments to format the plot:
            - figsize (tuple(int, int)): the figure size,
            - title (str): figure title,
            - xlabel (str): x-axis label,
            - ylabel (str): y-axis label,
            - logx (bool): if True set x-axis to logarithmic scale,
            - logy (bool): if True set y-axis to logarithmic scale,
            - fontsize (int): font size of label, title 15% larger, ticks 15% smaller.

    Returns:
        fig (figure object): the produced figure object.
        ax (ax object): the produced axis object.
    """

    if "figsize" not in kwargs:
        kwargs["figsize"] = (5, 5)
    if "title" not in kwargs:
        kwargs["title"] = "SOM"
    if "cbar_label" not in kwargs:
        kwargs["cbar_label"] = "Feature value"
    if "fontsize" not in kwargs:
        kwargs["fontsize"] = 12

    if ax is None:
        fig, ax_used = plt.subplots(figsize=kwargs["figsize"])
    elif isinstance(ax, Sequence):
        fig = ax[0].figure
        ax_used = ax[0]
    else:
        fig = ax.figure
        ax_used = ax

    ax_used = polygons_class.draw_map(fig, centers, feature,
                                      cmap=kwargs['cmap'] if 'cmap' in kwargs
                                      else plt.get_cmap('viridis'))
    ax_used.set_title(kwargs["title"], size=kwargs["fontsize"] * 1.15)

    divider = make_axes_locatable(ax_used)

    if not np.isnan(feature).all():
        cax = divider.append_axes("right", size="5%", pad=0.0)
        cbar = plt.colorbar(ax_used.collections[0], cax=cax)
        cbar.set_label(kwargs["cbar_label"], size=kwargs["fontsize"])
        cbar.ax.tick_params(labelsize=kwargs["fontsize"] * .85)
        cbar.outline.set_visible(False)

    fig.tight_layout()
    plt.sca(ax_used)

    if not file_name.endswith((".png", ".jpg", ".pdf")):
        file_name += ".png"

    if print_out:
        plt.savefig(file_name, bbox_inches="tight", dpi=300)
    if show:
        plt.show()

    return fig, ax_used
########################################################

def scatter_on_map(datagroups: Collection[np.ndarray], centers: Collection[np.ndarray],
                   polygons_class: Polygon,
                   color_val: bool = None,
                   show: bool = True, print_out: bool = False,
                   file_name: str = "./som_scatter.png",
                   **kwargs: Tuple[int]) -> Tuple[Figure, plt.Axes]:
    """Scatter plot with points projected onto a 2D SOM.

    Args:
        datagroups (list[array,...]): Coordinates of the projected points.
            This must be a nested list/array of arrays, where each element of 
            the list is a group that will be plotted separately.
        centers (list or array): The list of SOM nodes center point coordinates
            (e.g. node.pos)
        color_val (array): The feature value to use as color map, if None
                the map will be plotted as white.
        polygons_class (polygons): The polygons class carrying information on the
            map topology.
        show (bool): Choose to display the plot.
        print_out (bool): Choose to save the plot to a file.
        file_name (str): Name of the file where the plot will be saved if
            print_out is active. Must include the output path.
        kwargs (dict): Keyword arguments to format the plot:
            - figsize (tuple(int, int)): the figure size,
            - title (str): figure title,
            - cbar_label (str): colorbar label,
            - fontsize (int): font size of label, title 15% larger, ticks 15% smaller,
            - cmap (ListedColormap): a custom colormap.

    Returns:
        fig (figure object): the produced figure object.
        ax (ax object): the produced axis object.
    """

    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (5, 5)
    if "title" not in kwargs.keys():
        kwargs["title"] = "Projection onto SOM"
    if "fontsize" not in kwargs.keys():
        kwargs["fontsize"] = 12

    if color_val is None:
        color_val = np.full(len(centers), np.nan)

    fig, ax = plot_map(centers, color_val,
                       polygons_class,
                       show=False, print_out=False,
                       **kwargs)

    for i, group in enumerate(datagroups):
        print(group[:,0])
        ax.scatter(group[:, 0], group[:, 1],
                   color=Distinct20()[i % 20], edgecolor="#ffffff",
                   linewidth=1, label='{:d}'.format(i))

    plt.legend(bbox_to_anchor=(-.025, 1), fontsize=kwargs["fontsize"] * .85,
               frameon=False, title='Groups', ncol=int(len(datagroups) / 10.0) + 1,
               title_fontsize=kwargs["fontsize"])

    if not file_name.endswith((".png", ".jpg", ".pdf")):
        file_name += ".png"

    if print_out == True:
        plt.savefig(file_name, bbox_inches="tight", dpi=300)
    if show == True:
        plt.show()

    return fig, ax
