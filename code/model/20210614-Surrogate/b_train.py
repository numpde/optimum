# RA, 2021-06-14

"""
Train a surrogate model for transition times.
"""

VERSION = "v1"

import tensorflow as tf

import percache

from twig import log
import logging

log.parent.handlers[0].setLevel(logging.DEBUG)

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from tcga.utils import unlist1, relpath, mkdir, first, Now

from opt_utils.graph import largest_component, GraphNearestNode, GraphPathDist
from opt_utils.misc import Section

from opt_tf import FunctionalLogger, ConditionalAbort

BASE = Path(__file__).parent
DATA = next(p for p in BASE.parents for p in p.glob("**/data")).resolve()
AREA = "manhattan"
EDGE_WEIGHT_KEY = "len"  # TODO

cache = percache.Cache(str(mkdir(Path(__file__).parent / "cache") / "percache.dat"), livesync=True)
cache.clear(maxage=(60 * 60 * 24))

timestamp = Now()
out_dir = mkdir((Path(__file__).with_suffix('') / VERSION).resolve())


def load_graph(area: str):
    with Section("Loading graph", out=log.debug):
        file = max(DATA.glob(f"**/*graph/{area}.pkl"))
        log.debug(f"graph file: {relpath(file)}")

        with file.open(mode='rb') as fd:
            import pickle
            g = pickle.load(fd)

        assert (type(g) is nx.DiGraph), \
            f"{relpath(file)} has wrong type {type(g)}."

        return g


@cache
def make_df(n, seed=43, area=AREA, edge_weight_key=EDGE_WEIGHT_KEY) -> pd.DataFrame:
    # Note: prefer make_df(100) over make_df(n=100) because of @cache
    graph = largest_component(load_graph(area))

    rng = np.random.default_rng(seed=seed)
    edge_weight = nx.get_edge_attributes(graph, name=edge_weight_key)
    node_loc = nx.get_node_attributes(graph, name='loc')

    assert len(edge_weight)
    assert len(node_loc)

    pairs = rng.choice(graph.nodes, size=(n, 2), replace=True)
    lat_lon = [(*node_loc[a], *node_loc[b]) for (a, b) in pairs]
    arc_len = [nx.shortest_path_length(graph, a, b, weight=edge_weight_key) for (a, b) in pairs]

    return pd.DataFrame(lat_lon, columns=['xa_lat', 'xa_lon', 'xb_lat', 'xb_lon']).assign(y=arc_len)


def make_ds(n):
    # Note: ds_train and ds_valid should differ in `n`; hence `seed=n`.
    df = make_df(n, seed=n)

    ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(df.drop(columns='y').values),
        tf.data.Dataset.from_tensor_slices(df.y.values)
    ))

    return ds


def make_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=[4]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=32, activation='swish'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(units=16, activation='swish'),
        tf.keras.layers.Dense(units=8, activation='swish'),
        tf.keras.layers.Dense(units=1, activation='relu')
    ])

    opt = tf.keras.optimizers.RMSprop(learning_rate=1e-2)
    model.compile(optimizer=opt, loss='mae')

    return model


def main():
    ds_train = make_ds(n=10000)
    ds_valid = make_ds(n=1000)  # avoid using the same n
    log.debug(f"first of training dataset: \n{list(ds_train.take(1))}")
    log.debug(f"first of validation dataset: \n{list(ds_valid.take(1))}")

    # log.debug(f"{first(first(ds_train.take(1)))}")

    model = make_model()
    log.debug(f"prediction example: {model.predict(np.asarray([[40.74, -74, 40.7, -74]]))}")

    log_dir = (out_dir / f"history/{timestamp}")
    log.debug(f"log_dir = {relpath(log_dir)}")

    model.fit(
        ds_train.shuffle(2 ** 10).batch(64),
        epochs=1200,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=111, monitor='loss'),
            tf.keras.callbacks.ReduceLROnPlateau(patience=10, monitor='loss', factor=0.9, min_lr=1e-5, cooldown=5),
            FunctionalLogger({"lr": (lambda m: m.optimizer.lr)}),
            tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1),
            tf.keras.callbacks.ModelCheckpoint(log_dir / "model.tf", save_best_only=True, monitor='val_loss'),
        ],
        validation_data=ds_valid.batch(2 ** 10),
    )


if __name__ == '__main__':
    main()
