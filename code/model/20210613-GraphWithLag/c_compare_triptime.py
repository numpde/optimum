# RA, 2021-06-20

EDGE_TTT_KEY = "lag"

import pickle

import numpy as np
import pandas as pd
import networkx as nx
from sorcery import dict_of

from twig import log
from pathlib import Path
from datetime import timedelta, datetime

from percache import Cache as cache
from geopy.distance import distance as geodistance

from plox import Plox
from tcga.utils import unlist1, relpath, mkdir, first, Now, whatsmyname, First

from opt_trips.trips import get_raw_trips, KEEP_COLS, with_nearest_ingraph
from opt_utils.graph import largest_component, GraphNearestNode
from opt_utils.misc import Memo

DATA = next(p for p in Path(__file__).parents for p in p.glob("**/model")).resolve()

cache = cache(str(mkdir(Path(__file__).with_suffix('')) / f"percache.dat"), livesync=True)
cache.clear(maxage=int(timedelta(days=7).total_seconds()))


def load_graph(file):
    with file.open(mode='rb') as fd:
        return largest_component(pickle.load(fd))


def get_trips_times_square(*, graph_path, table_names, sql_where, sql_limit):
    f = (40.75798, -73.98550)  # Times Square
    r = 1000  # meters

    bb = pd.DataFrame(columns=['lat', 'lon', 'alt'], data=[
        geodistance(kilometers=(r * 1.01 / 1e3)).destination(f, bearing=bearing)
        for bearing in [-90, 0, 90, 180]
    ])

    sql_where = f"""
        ({min(bb.lat)} <= xa_lat) and (xa_lat <= {max(bb.lat)}) and
        ({min(bb.lon)} <= xa_lon) and (xa_lon <= {max(bb.lon)}) and
        ({sql_where})
    """

    trips = pd.concat(axis=0, objs=[
        get_raw_trips(table_name, where=sql_where, limit=sql_limit).assign(table_name=table_name)
        for table_name in sorted(table_names)
    ])

    trips = trips[list(KEEP_COLS)].sort_values(by='ta')
    trips = trips.assign(xa=list(zip(trips.xa_lat, trips.xa_lon)))
    trips = trips.assign(xb=list(zip(trips.xb_lat, trips.xb_lon)))

    trips = trips[trips[['xa', 'xb']].applymap(lambda x: (geodistance(x, f).m <= r)).all(axis=1)]

    graph = load_graph(graph_path)
    trips = with_nearest_ingraph(trips, graph)

    trips['dt_rep'] = (trips.tb - trips.ta).dt.total_seconds()
    trips['dt_est'] = [nx.shortest_path_length(graph, *ab, weight=EDGE_TTT_KEY) for ab in zip(trips.ia, trips.ib)]

    return (graph, trips)


def hist(graph, trips):
    with Plox() as px:
        from scipy.stats import gaussian_kde

        # tt = np.linspace(0, max(trips.quantile(axis=0, q=0.99)), 101)
        tt = np.linspace(0, timedelta(minutes=10).total_seconds(), 101)
        rep = gaussian_kde(trips.dt_rep)(tt)
        est = gaussian_kde(trips.dt_est)(tt)

        px.a.plot(tt, rep, label="Reported")
        px.a.plot(tt, est, label="Model")

        px.a.legend()
        px.show()


def main():
    for hour in [6, 18]:
        memo = Memo()
        hour = memo(hour)
        setup = memo({
            'table_names': {"green_tripdata_2016-05", "yellow_tripdata_2016-05"},
            'sql_where': f"(({hour}) == cast(strftime('%H', [ta]) as int))",
            'sql_limit': 111111,
            'graph_path': max(DATA.glob(f"*WithLag/*train/**/lag/H={hour}/**/manhattan.pkl")),
        })

        log.info(memo)

        (graph, trips) = get_trips_times_square(**setup)

        hist(graph, trips)


if __name__ == '__main__':
    main()
