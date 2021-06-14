# RA, 2021-06-14

import tensorflow as tf

import contextlib

import matplotlib.pyplot as plt
import percache

from more_itertools import pairwise
from tensorflow.python.data import AUTOTUNE

from twig import log
import logging

log.parent.handlers[0].setLevel(logging.DEBUG)

from pathlib import Path

from plox import Plox
from tcga.utils import unlist1, relpath, mkdir, first, Now, whatsmyname

from opt_maps import maps
from opt_utils.graph import largest_component, GraphNearestNode, GraphPathDist
from opt_utils.style import default_style, name2color, name2cmap, get_velocity_cmap
from opt_utils.misc import Section

from b_train import make_df, load_graph, AREA, BASE, DATA

out_dir = mkdir(Path(__file__).with_suffix(''))


def load_model():
    path_to_model = max(BASE.glob("**/v*/history/*/model.tf"))
    log.info(f"Loading model from: {relpath(path_to_model)}")

    model = tf.keras.models.load_model(path_to_model)

    return model


def estimated_vs_reference(df):
    with Plox() as px:
        px.a.scatter(df.y * 1e-3, df.p * 1e-3, s=20, c='C3', alpha=0.3, edgecolor='none')
        px.a.set_xlabel("Reference, km")
        px.a.set_ylabel("Estimated, km")
        (mi, ma) = (0, max([*px.a.get_xlim(), *px.a.get_ylim()]))
        px.a.plot([mi, ma], [mi, ma], '--', lw=1, c='k')
        px.a.set_xlim(mi, ma)
        px.a.set_ylim(mi, ma)
        px.a.axis('square')
        px.f.savefig(out_dir / f"{whatsmyname()}.png")


def visualize_model(model):
    with (out_dir / "model.txt").open(mode='w') as fd:
        with contextlib.redirect_stdout(fd):
            model.summary()

    tf.keras.utils.plot_model(
        model,
        to_file=(out_dir / f"model.png"),
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

    visualize_model(model)
    estimated_vs_reference(df)


if __name__ == '__main__':
    main()
