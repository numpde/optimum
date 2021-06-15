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
from tcga.utils import unlist1, relpath, mkdir, first

# from opt_maps import maps
from opt_utils.graph import largest_component, GraphNearestNode, GraphPathDist
from opt_utils.style import default_style, name2color, name2cmap, get_velocity_cmap
from opt_utils.misc import Section
from opt_trips.trips import get_trips_mit_alles

BASE = Path(__file__).parent
DATA = next(p for p in BASE.parents for p in p.glob("**/data")).resolve()

out_dir = mkdir(Path(__file__).with_suffix(''))

parallel_map = map

area = "manhattan"
table_names = sorted({"green_tripdata_2016-05", "yellow_tripdata_2016-05"})

rng = np.random.default_rng(seed=43)


def plot3_day1():
    where = "('2016-05-01 00:00' <= ta) and (tb <= '2016-05-02 00:00')"
    data = get_trips_mit_alles("manhattan", table_names, where=where)

    graph: nx.DiGraph = data.graph
    trips: pd.DataFrame = data.trips

    edge_time_weight = 'lag'

    edge_len = nx.get_edge_attributes(graph, name='len')
    edge_lag = nx.get_edge_attributes(graph, name=edge_time_weight)

    if not edge_lag:
        log.warning(f"Imputing edge `lag` from `len`.")
        speed = 6  # m/s
        edge_lag = {e: timedelta(seconds=(v / speed)) for (e, v) in edge_len.items()}

    node_loc = pd.DataFrame(nx.get_node_attributes(graph, name='loc'), index=["lat", "lon"])

    t0 = min(trips.ta)
    t1 = t0 + timedelta(hours=1)

    trips = trips[(t0 <= trips.ta) & (trips.tb <= t1)]

    # Single-passenger trips only
    trips = trips[trips.n == 1]

    log.info(f"Got {len(trips)} trips (between {t0} and {t1}).")

    import plotly.graph_objects as go
    import plotly.express as px

    trips = trips.sample(n=333, random_state=43)

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
                z=(path.lag - t0).dt.total_seconds() / 60,
                # marker=dict(size=0),
                line=dict(width=1, color=color),
                mode="lines",
            ),
        )

        fig.update_layout(showlegend=False)

    fig.show()




def main():
    plot3_day1()


if __name__ == '__main__':
    main()
