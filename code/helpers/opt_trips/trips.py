# RA, 2021-06-14


from typing import List

import percache

from twig import log
import logging

log.parent.handlers[0].setLevel(logging.DEBUG)

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from tcga.utils import unlist1, relpath, mkdir, first

from opt_utils.graph import largest_component, GraphNearestNode, GraphPathDist
from opt_utils.misc import Section

BASE = Path(__file__).parent
DATA = next(p for p in BASE.parents for p in p.glob("**/data")).resolve()

KEEP_COLS = ['ta', 'tb', 'xa_lon', 'xa_lat', 'xb_lon', 'xb_lat', 'n', 'distance', 'duration', 'table_name']

parallel_map = map

cache = percache.Cache(str(mkdir(Path(__file__).parent / "cache") / "percache.dat"), livesync=True)
cache.clear(maxage=(60 * 60 * 24 * 7))


def query_trips(sql):
    import sqlite3
    return pd.read_sql_query(
        sql=sql,
        con=sqlite3.connect(
            unlist1(DATA.glob("**/*NYCTLC/**/trips.db"))
        ),
        parse_dates=["ta", "tb"],
    )


def get_road_graph(area):
    file = unlist1(DATA.glob(f"**/*OSM/*graph/{area}.pkl"))

    with file.open(mode='rb') as fd:
        import pickle
        g = pickle.load(fd)

    assert (type(g) is nx.DiGraph), \
        f"{relpath(file)} has wrong type {type(g)}."

    return g


def get_raw_trips(table_name, where="", order="random()", limit=11111) -> pd.DataFrame:
    with Section("Querying trips", out=log.debug):
        sql = " ".join([
            f"SELECT * FROM [{table_name}]",
            f"WHERE    ({where}) " if where else "",
            f"ORDER BY ({order}) " if order else "",
            f"LIMIT    ({limit}) " if limit else "",
        ])

        df = query_trips(sql)
        df = df.rename(columns={'passenger_count': "n"})

        return df


def with_nearest_ingraph(trips, graph):
    with Section("Computing nearest in-graph nodes", out=log.debug):
        nearest_node = GraphNearestNode(graph)

        # (index, values) correspond to (graph node id, distance)
        A = nearest_node(list(zip(trips.xa_lat, trips.xa_lon)))
        B = nearest_node(list(zip(trips.xb_lat, trips.xb_lon)))

        # Attach in-graph node estimates of pickup and dropoff
        trips['ia'] = A.index
        trips['ib'] = B.index

        # Grace distance from given lat/lon to nearest in-graph lat/lon
        MAX_NEAREST = 20  # meters
        ii = (A.values <= MAX_NEAREST) & (B.values <= MAX_NEAREST)
        trips = trips.loc[ii]

        log.debug(f"Keep {sum(ii)}, drop {sum(~ii)}.")

        return trips


def with_shortest_distance(trips, graph, edge_weight="len"):
    with Section("Computing shortest distances", out=log.debug):
        return trips.join(
            pd.DataFrame(
                data=parallel_map(GraphPathDist(graph, edge_weight=edge_weight), zip(trips.ia, trips.ib)),
                index=trips.index,
                columns=['path', 'shortest'],
            )
        )


@cache
def get_trips_mit_alles(area: str, table_names: List[str], keep_cols=tuple(KEEP_COLS), **kwargs):
    trips = pd.concat(axis=0, objs=[
        get_raw_trips(table_name, **kwargs).assign(table_name=table_name)
        for table_name in sorted(table_names)
    ])

    trips = trips[list(keep_cols)]

    trips = trips.sort_values(by='ta')

    log.debug(f"Trips: \n{trips.head(6).to_markdown()}")

    graph = largest_component(get_road_graph(area))

    trips = with_nearest_ingraph(trips, graph)
    trips = with_shortest_distance(trips, graph)

    return pd.Series({'area': area, 'table_names': table_names, 'graph': graph, 'trips': trips})


if __name__ == '__main__':
    table_names = {"green_tripdata_2016-05", "yellow_tripdata_2016-05"}
    data = get_trips_mit_alles("manhattan", sorted(table_names))

    log.info(f"Got data: \n{data}")
