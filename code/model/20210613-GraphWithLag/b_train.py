# RA, 2021-06-15

"""
Estimate travel time across graph edges.
"""

VERSION = "v1"

import shutil
import pickle
import percache

import twig
from twig import log
import logging

from typing import List
from pathlib import Path
from more_itertools import pairwise

import numpy as np
import pandas as pd
import networkx as nx

import scipy.sparse

from tcga.utils import unlist1, relpath, mkdir, first, Now, whatsmyname

from opt_utils.graph import largest_component, GraphNearestNode, GraphPathDist
from opt_utils.misc import Section
from opt_trips.trips import with_nearest_ingraph, get_raw_trips, KEEP_COLS

from z_utils import plot_graph_velocity

BASE = Path(__file__).parent
DATA = next(p for p in BASE.parents for p in p.glob("**/data")).resolve()
AREA = "manhattan"

# Graph edge key for time-to-transition
EDGE_TTT_KEY = "lag"  #
assert not (EDGE_TTT_KEY == 'len'), "This doesn't make sense."

#
# log.parent.handlers[0].setLevel(logging.DEBUG)

# A constant velocity prior for the initial estimate of edge time-to-transition ("lag")
PRIOR_VEL = 5  # m/s

# Hour of the day to focus on (e.g. 18)
HOUR = 18

out_dir = mkdir((Path(__file__).with_suffix('') / f"{VERSION}/{EDGE_TTT_KEY}/H={HOUR}/{Now()}").resolve())

parallel_map = map

rng = np.random.default_rng(seed=43)


def graph_filename(area: str) -> Path:
    return max(DATA.glob(f"**/*graph/{area}.pkl"))


def load_graph(file: Path):
    with Section("Loading graph", out=log.debug):
        log.info(f"graph file: {relpath(file)}")

        with file.open(mode='rb') as fd:
            import pickle
            g = pickle.load(fd)

        assert (type(g) is nx.DiGraph), \
            f"{relpath(file)} has wrong type {type(g)}."

        if not nx.get_edge_attributes(g, name=EDGE_TTT_KEY):
            log.info(f"Imputing edge time-to-transition.")
            edge_len = nx.get_edge_attributes(g, name='len')
            assert edge_len
            nx.set_edge_attributes(g, {e: (d / PRIOR_VEL) for (e, d) in edge_len.items()}, name=EDGE_TTT_KEY)

        return largest_component(g)


def get_trips(table_names: List[str], **kwargs):
    trips = pd.concat(axis=0, objs=[
        get_raw_trips(table_name, **kwargs).assign(table_name=table_name)
        for table_name in sorted(table_names)
    ])

    trips = trips[list(KEEP_COLS)]
    trips = trips.sort_values(by='ta')

    log.debug(f"Trips: \n{trips.head(3).to_markdown()}\n...")

    return trips


def get_trips_by_hour(table_names: List[str], hour=8):
    where = f"(({hour}) == cast(strftime('%H', [ta]) as int))"
    trips = get_trips(table_names, where=where)
    # trips = trips.rename(columns={'distance': "distance_ref"})
    return trips


def with_ingraph_trajectories(trips, graph, edge_weight=EDGE_TTT_KEY):
    with Section(f"Computing in-graph trajectories for `{edge_weight}`", out=log.debug):
        try:
            cols = [f"{edge_weight}_path", f"{edge_weight}"]
            trips = trips.drop(columns=cols)
        except KeyError:
            pass

        return trips.join(
            pd.DataFrame(
                data=parallel_map(GraphPathDist(graph, edge_weight=edge_weight), zip(trips.ia, trips.ib)),
                index=trips.index,
                columns=cols,
            )
        )


def trip_x_edge_matrix(paths, graph):
    e2j = {e: j for (j, e) in enumerate(graph.edges)}

    data = pd.DataFrame(
        data=[(t, e2j[e]) for (t, path) in enumerate(paths) for e in pairwise(path)],
        columns=["trip_n", "edge_j"],
    )

    return scipy.sparse.coo_matrix((np.ones(len(data)), (data.trip_n, data.edge_j)), shape=(len(paths), len(e2j)))


def edge_vec(graph: nx.DiGraph, edge_weight='len'):
    return np.array([d for (a, b, d) in graph.edges.data(edge_weight)])


def train_iteration(graph: nx.DiGraph, trips: pd.DataFrame):
    graph = graph.copy()

    # Plausibility bounds on transition times
    (te_min, te_max) = np.outer([0.1, 10], edge_vec(graph, 'len') / PRIOR_VEL)

    # Attach in-graph trajectories (shortest for the `edge_weight`)
    trips_batch = with_ingraph_trajectories(trips, graph, edge_weight=EDGE_TTT_KEY)

    # Trips x Edges incidence matrix
    TE = trip_x_edge_matrix(trips_batch[f"{EDGE_TTT_KEY}_path"], graph)

    # Current edge transition times
    te = edge_vec(graph, EDGE_TTT_KEY)

    # Reported trip durations
    tt = np.array(trips_batch.duration)

    alpha = 0.1
    for _ in range(33):
        grad = (TE.T @ (TE @ te - tt)) / len(tt)
        te = np.clip(te - alpha * grad, a_min=(0.9 * te), a_max=(1.1 * te))
        te = np.clip(te, a_min=te_min, a_max=te_max)

    # Correlation between reported and predicted durations
    corr = np.min(np.corrcoef(tt, TE @ te))
    log.info(f"Correlation of reported vs estimated duration: {corr}.")

    nx.set_edge_attributes(graph, dict(zip(graph.edges, te)), name=EDGE_TTT_KEY)

    return graph


def train(graph, trips):
    # trips = trips.sample(n=27, random_state=101)

    trips = with_nearest_ingraph(trips, graph)

    # Keep the trips where reported and plausible estimated travel distances are similar
    trips = with_ingraph_trajectories(trips, graph, edge_weight='len')
    anomaly_score = (lambda x, y: np.exp(np.abs(np.log(x / y))))
    trips = trips.assign(anomaly=anomaly_score(trips.len, trips.distance))
    trips = trips[trips.anomaly <= (1.20)]

    log.info(f"Number of trips for training: {len(trips)}.")
    log.debug(f"E.g.: \n{trips.head(3)}")

    log_dir = mkdir(out_dir / f"{whatsmyname()}")
    log.info(f"log_dir = {relpath(log_dir)}")

    for i in range(10):
        try:
            graph = train_iteration(graph, trips)
            yield graph

            with plot_graph_velocity(graph, edge_weight=EDGE_TTT_KEY) as px:
                px.f.savefig(log_dir / f"velocity_i={i}.png")
        except KeyboardInterrupt:
            break


def main():
    area = AREA
    table_names = sorted({"green_tripdata_2016-05", "yellow_tripdata_2016-05"})

    graph_file = graph_filename(area)
    graph = load_graph(graph_file)

    trips = get_trips_by_hour(table_names, hour=HOUR)

    for graph in train(graph, trips):
        with (out_dir / graph_file.name).open(mode='wb') as fd:
            pickle.dump(graph, fd)


if __name__ == '__main__':
    try:
        main()
    finally:
        shutil.copyfile(twig.LOG_FILE, out_dir / twig.LOG_FILE.name)
