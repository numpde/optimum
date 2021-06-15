# RA, 2021-06-15

import contextlib
import logging

import matplotlib.pyplot as plt
import percache

from more_itertools import pairwise

from twig import log
from inclusive import range

log.parent.handlers[0].setLevel(logging.DEBUG)

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from plox import Plox
from tcga.utils import unlist1, relpath, mkdir, first

from opt_maps import maps
from opt_utils.graph import largest_component, GraphNearestNode, GraphPathDist
from opt_utils.style import default_style, name2color, name2cmap, get_velocity_cmap
from opt_utils.misc import Section

BASE = Path(__file__).parent
DATA = next(p for p in BASE.parents for p in p.glob("**/data")).resolve()

out_dir = mkdir(Path(__file__).with_suffix(''))

parallel_map = map



@contextlib.contextmanager
def plot_graph_velocity(graph: nx.DiGraph, edge_weight='len') -> Plox:
    nodes = pd.DataFrame(data=nx.get_node_attributes(graph, "loc"), index=["lat", "lon"]).T

    edge_len = nx.get_edge_attributes(graph, name='len')
    edge_lag = nx.get_edge_attributes(graph, name=edge_weight)

    with Section("Getting the background OSM map", out=log.debug):
        extent = maps.ax4(nodes.lat, nodes.lon)
        background = maps.get_map_by_bbox(maps.ax2mb(*extent))

    with Section("Computing edge velocities", out=log.debug):
        edge_vel = pd.Series({e: edge_len[e] / edge_lag[e] for e in graph.edges})

    with Plox({**default_style, 'font.size': 5}) as px:
        px.a.imshow(background, extent=extent, interpolation='quadric', zorder=-100)

        px.a.axis("off")

        px.a.set_xlim(extent[0:2])
        px.a.set_ylim(extent[2:4])

        (vmin, vmax) = (1, 7)
        edge_vel: pd.Series = edge_vel.clip(lower=vmin, upper=vmax)

        nx.draw_networkx_edges(
            graph.edge_subgraph(edge_vel.index),
            ax=px.a,
            pos=nx.get_node_attributes(graph, name="pos"),
            edgelist=edge_vel.index,
            edge_color=edge_vel,
            edge_cmap=get_velocity_cmap(),
            edge_vmin=vmin, edge_vmax=vmax,
            arrows=False, node_size=0, alpha=0.8, width=0.3,
        )

        import matplotlib.cm
        import matplotlib.colors
        norm = matplotlib.colors.Normalize(vmin=edge_vel.min(), vmax=edge_vel.max())
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=get_velocity_cmap())
        sm.set_clim(vmin, vmax)

        from matplotlib.transforms import Bbox
        ((x0, y0), (x1, y1)) = px.a.get_position().get_points()
        cax: plt.Axes = px.f.add_axes(Bbox(((x0 + 0.01, y1 - 0.07), (x0 + (x1 - x0) * 0.5, y1 - 0.05))))
        cax.set_title("Est. velocity, m/s")
        ticks = range[vmin + 1, vmax - 1]
        px.f.colorbar(mappable=sm, cax=cax, orientation='horizontal', ticks=ticks)

        yield px
