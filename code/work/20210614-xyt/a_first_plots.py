# RA, 2021-06-14


import contextlib
import logging

from typing import List
from datetime import datetime, timedelta

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
from tcga.utils import unlist1, relpath, mkdir, first, whatsmyname

# from opt_maps import maps
from opt_utils.graph import largest_component, GraphNearestNode, GraphPathDist
from opt_utils.style import default_style, name2color, name2cmap, get_velocity_cmap
from opt_utils.misc import Section
from opt_trips.trips import get_trips_mit_alles, get_raw_trips, KEEP_COLS, with_nearest_ingraph, with_shortest_distance

BASE = Path(__file__).parent
DATA = next(p for p in BASE.parents for p in p.glob("**/model")).resolve()
EDGE_TTT_KEY = 'lag'  # time-to-transition

graph_source = max(DATA.glob(f"*WithLag/*train/**/lag/H=18"))

out_dir = mkdir(Path(__file__).with_suffix(''))

cache = percache.Cache(str(out_dir / f"percache.dat"), livesync=True)
cache.clear(maxage=(60 * 60 * 24 * 7))

parallel_map = map

area = "manhattan"
table_names = sorted({"green_tripdata_2016-05", "yellow_tripdata_2016-05"})

rng = np.random.default_rng(seed=43)


def load_graph(area: str):
    with Section("Loading graph", out=log.debug):
        file = max(graph_source.glob(f"**/{area}.pkl"))
        log.debug(f"graph file: {relpath(file)}")

        with file.open(mode='rb') as fd:
            import pickle
            g = pickle.load(fd)

        assert (type(g) is nx.DiGraph), \
            f"{relpath(file)} has wrong type {type(g)}."

        assert nx.get_edge_attributes(g, name=EDGE_TTT_KEY)

        return g


@cache
def get_trips(area: str, table_names: List[str], **kwargs):
    trips = pd.concat(axis=0, objs=[
        get_raw_trips(table_name, **kwargs).assign(table_name=table_name)
        for table_name in sorted(table_names)
    ])

    trips = trips[list(KEEP_COLS)]

    trips = trips.sort_values(by='ta')

    log.debug(f"Trips: \n{trips.head(3).to_markdown()} \netc.")

    graph = largest_component(load_graph(area))

    trips = with_nearest_ingraph(trips, graph)
    trips = with_shortest_distance(trips, graph)

    return pd.Series({'area': area, 'table_names': table_names, 'graph': graph, 'trips': trips})


def morning_of_day1():
    hour = "06"
    where = f"('2016-05-01 {hour}:00' <= ta) and (tb <= '2016-05-02 {hour}:00')"
    data = get_trips("manhattan", table_names, where=where)

    graph: nx.DiGraph = data.graph
    trips: pd.DataFrame = data.trips

    edge_time_weight = EDGE_TTT_KEY

    edge_len = nx.get_edge_attributes(graph, name='len')
    edge_lag = nx.get_edge_attributes(graph, name=edge_time_weight)

    if not edge_lag:
        log.warning(f"Imputing edge `lag` from `len`.")
        speed = 6  # m/s
        edge_lag = {e: timedelta(seconds=(d / speed)) for (e, d) in edge_len.items()}
    else:
        # Convert float (seconds) to `timedelta`
        edge_lag = {e: timedelta(seconds=t) for (e, t) in edge_lag.items()}

    node_loc = pd.DataFrame(nx.get_node_attributes(graph, name='loc'), index=["lat", "lon"])

    # Further subset to the first hour
    t0 = min(trips.ta)
    t1 = t0 + timedelta(hours=1)
    trips = trips[(t0 <= trips.ta) & (trips.tb <= t1)]

    # Single-passenger trips only
    trips = trips[trips.n == 1]

    log.info(f"Got {len(trips)} trips (between {t0} and {t1}).")

    import plotly.graph_objects as go
    import plotly.express as px

    trips = trips.sample(n=100, random_state=43)

    log.info(f"Subsampled to {len(trips)} trips.")

    fig = go.Figure()

    for (i, trip) in trips.iterrows():
        path = nx.shortest_path(graph, trip.ia, trip.ib, weight=edge_time_weight)
        path = node_loc[path].T
        path = path.assign(lag=(np.cumsum([trip.ta] + [edge_lag[e] for e in pairwise(path.index)])))
        path = path.assign(len=(np.cumsum([0] + [edge_len[e] for e in pairwise(path.index)])))

        # Distance in kilometers
        path.len = path.len / 1e3

        # log.debug(f"path: \n{path}")

        color = rng.choice(px.colors.sequential.Plasma)

        fig.add_trace(
            go.Scatter3d(
                x=path.lon,
                y=path.lat,
                # z=path.lag,
                z=(path.lag - t0).dt.total_seconds() / 60,
                # marker=dict(size=0),
                line=dict(width=1, color=color),
                mode="lines",
            ),
        )

    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis_title="lon",
            yaxis_title="lat",
            zaxis_title=f"{hour}h + minutes",
        ),
    )

    fig.write_html(str(out_dir / f"{whatsmyname()}.html"))


def main():
    morning_of_day1()


if __name__ == '__main__':
    main()
