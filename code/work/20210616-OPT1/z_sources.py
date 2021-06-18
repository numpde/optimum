# RA, 2021-06-18

EDGE_TTT_KEY = "lag"  # time-to-transition attribute

import numpy as np
import pandas as pd
import networkx as nx

from twig import log
from pathlib import Path
from datetime import timedelta, datetime

from geopy.distance import distance as geodistance

from tcga.utils import unlist1, relpath, mkdir, first, Now, whatsmyname, First

from opt_utils.graph import largest_component, GraphNearestNode
from opt_trips.trips import get_raw_trips, KEEP_COLS, with_nearest_ingraph

DATA = next(p for p in Path(__file__).parents for p in p.glob("**/model")).resolve()

graph_source = max(DATA.glob(f"*WithLag/*train/**/lag/H=18"))


def load_graph(area):
    file = max(graph_source.glob(f"**/{area}.pkl"))
    log.debug(f"Graph file: {relpath(file)}")

    with file.open(mode='rb') as fd:
        import pickle
        g = pickle.load(fd)

    assert (type(g) is nx.DiGraph), \
        f"{relpath(file)} has wrong type {type(g)}."

    assert nx.get_edge_attributes(g, name=EDGE_TTT_KEY)

    return largest_component(g)


def attach_timewindows(trips: pd.DataFrame):
    (a_early, a_late, b_late) = (timedelta(minutes=2), timedelta(minutes=5), timedelta(minutes=10))
    trips = trips.assign(twa=trips.apply(axis=1, func=(lambda r: (r.ta - a_early, r.ta + a_late))))
    trips = trips.assign(twb=trips.apply(axis=1, func=(lambda r: (r.ta - a_early, r.tb + b_late))))
    return trips


def concentrated_subset(trips: pd.DataFrame, focal_point, focus_radius):
    dist_to_focal_point = pd.Series(index=trips.index, data=[
        max(geodistance(row.xa, focal_point).m, geodistance(row.xb, focal_point).m)
        for (i, row) in trips.iterrows()
    ])
    return trips[dist_to_focal_point <= focus_radius]


def get_problem_data(area, table_names: list, sql_where: str, max_trips: int, focal_point: tuple, focus_radius: float):
    trips = pd.concat(axis=0, objs=[
        get_raw_trips(table_name, where=sql_where).assign(table_name=table_name)
        for table_name in sorted(table_names)
    ])

    assert len(trips), \
        f"Query returned zero trips. Maybe a misshappen `where`: \n{sql_where}"

    log.debug(f"Raw query returned {len(trips)} trips.")

    trips = trips[list(KEEP_COLS)].sort_values(by='ta')
    trips = trips.assign(xa=list(zip(trips.xa_lat, trips.xa_lon)))
    trips = trips.assign(xb=list(zip(trips.xb_lat, trips.xb_lon)))

    graph = load_graph(area)
    trips = with_nearest_ingraph(trips, graph)
    trips = attach_timewindows(trips)
    trips = concentrated_subset(trips, focal_point=focal_point, focus_radius=focus_radius)

    depot = unlist1(GraphNearestNode(graph)([focal_point]).index)
    assert all(np.isclose(focal_point, graph.nodes[depot]['loc'], rtol=1e-3))

    trips = trips.head(max_trips)

    log.info(f"Filtered down to {len(trips)} trips: \n{trips.head(3).to_markdown()} \netc...")

    return pd.Series({'area': area, 'table_names': table_names, 'graph': graph, 'trips': trips, 'depot': depot})
