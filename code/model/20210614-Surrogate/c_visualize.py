# RA, 2021-06-14

import pandas as pd
import networkx as nx
import tensorflow as tf

import contextlib

from twig import log
import logging

log.parent.handlers[0].setLevel(logging.DEBUG)

from pathlib import Path

from plox import Plox
from tcga.utils import unlist1, relpath, mkdir, first, Now, whatsmyname

from opt_maps import maps
from opt_utils.graph import largest_component, GraphNearestNode, GraphPathDist
from opt_utils.style import default_style
from opt_utils.misc import Section

from b_train import make_df, load_graph, AREA, BASE, DATA, EDGE_WEIGHT_KEY

out_dir = mkdir(Path(__file__).with_suffix(''))


def load_model():
    path_to_model = max(BASE.glob("**/v*/history/*/model.tf"))
    log.info(f"Loading model from: {relpath(path_to_model)}")

    model = tf.keras.models.load_model(path_to_model)

    return model


def outlier_trajectories(df: pd.DataFrame):
    df = df.assign(d=(df.y - df.p).abs())
    df = df.nlargest(n=7, columns=['d'])

    from b_train import load_graph
    graph = load_graph(AREA)

    with GraphNearestNode(graph) as gnn:
        df = df.assign(ia=gnn(df[['xa_lat', 'xa_lon']].values).index, ib=gnn(df[['xb_lat', 'xb_lon']].values).index)

    with (out_dir / f"{whatsmyname()}.txt").open(mode='w') as fd:
        with contextlib.redirect_stdout(fd):
            print(df.to_markdown())

    with GraphPathDist(graph, edge_weight=EDGE_WEIGHT_KEY) as gpd:
        trajectories = list(map(gpd.path_only, zip(df.ia, df.ib)))

    nodes = pd.DataFrame(data=nx.get_node_attributes(graph, "loc"), index=["lat", "lon"]).T

    extent = maps.ax4(nodes.lat, nodes.lon)
    background = maps.get_map_by_bbox(maps.ax2mb(*extent))

    with Plox({**default_style, 'font.size': 5}) as px:
        px.a.imshow(background, extent=extent, interpolation='quadric', zorder=-100)
        px.a.axis("off")

        px.a.set_xlim(extent[0:2])
        px.a.set_ylim(extent[2:4])

        for traj in trajectories:
            (lat, lon) = nodes.loc[list(traj), ['lat', 'lon']].values.T
            px.a.plot(lon, lat, alpha=0.8, lw=1)

        px.f.savefig(out_dir / f"{whatsmyname()}.png")


def estimated_vs_reference(df: pd.DataFrame):
    with Plox() as px:
        px.a.scatter((df.y * 1e-3), (df.p * 1e-3), s=20, c='C3', alpha=0.3, edgecolor='none')
        px.a.set_xlabel("Reference, km")
        px.a.set_ylabel("Estimated, km")
        (mi, ma) = (0, max([*px.a.get_xlim(), *px.a.get_ylim()]))
        px.a.plot([mi, ma], [mi, ma], '--', lw=1, c='k')
        px.a.set_xlim(mi, ma)
        px.a.set_ylim(mi, ma)
        px.a.axis('square')
        px.f.savefig(out_dir / f"{whatsmyname()}.png")

    with (out_dir / f"{whatsmyname()}.txt").open(mode='w') as fd:
        with contextlib.redirect_stdout(fd):
            print(df.describe().to_markdown())


def visualize_model(model):
    with (out_dir / f"{whatsmyname()}.txt").open(mode='w') as fd:
        with contextlib.redirect_stdout(fd):
            model.summary()

    tf.keras.utils.plot_model(
        model,
        to_file=(out_dir / f"{whatsmyname()}.png"),
        show_shapes=True,
        show_layer_names=False,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )


def main():
    model = load_model()

    df = make_df(1000, seed=1000)
    df = df.assign(p=model.predict(df.drop(columns='y')))

    outlier_trajectories(df)
    visualize_model(model)
    estimated_vs_reference(df)


if __name__ == '__main__':
    main()
