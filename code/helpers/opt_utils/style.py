# RA, 2021-06-12

from matplotlib.colors import LinearSegmentedColormap
from plox import rcParam

default_style = {
    rcParam.Text.usetex: True,

    rcParam.Font.size: 14,
    rcParam.Axes.labelsize: "large",

    rcParam.Xtick.Major.size: 2,
    rcParam.Ytick.Major.size: 0,

    rcParam.Xtick.Major.pad: 1,
    rcParam.Ytick.Major.pad: 1,
}


def name2color(table_name):
    if "green" in table_name: return "green"
    if "yello" in table_name: return "darkorange"
    raise ValueError(f"Don't have a color for {table_name}.")


def name2cmap(table_name):
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list('company', ["white", name2color(table_name)])


def get_velocity_cmap():
    return LinearSegmentedColormap.from_list(name="noname", colors=["brown", "r", "orange", "g"])
