# RA, 2021-06-23

from twig import log

import contextlib

from typing import Iterable
from pathlib import Path
from datetime import timedelta
from more_itertools import pairwise

from sorcery import unpack_keys as unpack

import numpy as np
import pandas as pd
import networkx as nx
from z_sources import get_problem_data, preprocess_problem_data, read_subcase


def paths_of_route(route, graph, edge_ttt_weight):
    node_loc = pd.DataFrame(nx.get_node_attributes(graph, name='loc'), index=["lat", "lon"])

    edge_lag = nx.get_edge_attributes(graph, name=edge_ttt_weight)

    assert edge_lag, \
        f"Edge attribute {edge_ttt_weight} does not exist in the graph."

    for (uv, (ta, tb)) in zip(pairwise(route.i), pairwise(route.est_time_arr)):
        path = nx.shortest_path(graph, *uv, weight=edge_ttt_weight)
        path = node_loc[list(path)].T
        path = path.assign(lag=(np.cumsum([ta] + [timedelta(seconds=1) * edge_lag[e] for e in pairwise(path.index)])))

        # fix arrival at last node at time `tb`
        path.lag += (tb - max(path.lag))

        waiting_time = min(path.lag) - ta
        if (waiting_time.total_seconds() < -5):
            # presumed departure way before reported arrival at first node
            log.warning(
                f"Transit {uv[0]} -> {uv[1]} is reported to start after {ta} "
                f"but needs to start at {min(path.lag)} to arrive "
                f"by the reported time {tb}."
            )

        yield path


def infer_full_routes(routes: pd.DataFrame, graph, edge_ttt_weight) -> dict:
    return {
        iv: pd.concat(axis=0, objs=list(paths_of_route(route, graph, edge_ttt_weight=edge_ttt_weight)))
        for (iv, route) in routes.groupby(by='iv')
    }


@contextlib.contextmanager
def infer_trajectories(trips: pd.DataFrame, routes: pd.DataFrame, graph, edge_ttt_weight) -> Iterable[dict]:
    assert set(trips.iv).issubset(routes.iv)

    routes = infer_full_routes(routes, graph, edge_ttt_weight=edge_ttt_weight)

    for (i, trip) in trips.iterrows():
        if not pd.isna(trip.iv):
            route = routes[trip.iv]
            yield {'trip': i, 'traj': routes[(trip.iv_ta < route.lag) & (route.lag <= trip.iv_tb)].i}
        else:
            yield {'trip': i}


def main():
    for subcase_path in Path(__file__).parent.glob("c_grid_study*/*/*cases/*"):
        (params, routes, trips) = unpack(read_subcase(subcase_path))
        raise NotImplementedError


if __name__ == '__main__':
    main()
