# RA, 2021-06-20


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

from tcga.utils import unlist1, relpath, mkdir, first, Now, whatsmyname, First

from opt_trips.trips import get_raw_trips, KEEP_COLS, with_nearest_ingraph
from opt_utils.graph import largest_component, GraphNearestNode
from opt_utils.misc import Memo

DATA = next(p for p in Path(__file__).parents for p in p.glob("**/model")).resolve()

cache = cache(str(mkdir(Path(__file__).with_suffix('')) / f"percache.dat"), livesync=True)
cache.clear(maxage=timedelta(days=7).total_seconds())


def load_graph(file):
    with file.open(mode='rb') as fd:
        return largest_component(pickle.load(fd))


@cache
def get_trips_times_square(*, table_names, sql_where, sql_limit):
    trips = pd.concat(axis=0, objs=[
        get_raw_trips(table_name, where=sql_where, limit=sql_limit).assign(table_name=table_name)
        for table_name in sorted(table_names)
    ])

    trips = trips[list(KEEP_COLS)].sort_values(by='ta')
    trips = trips.assign(xa=list(zip(trips.xa_lat, trips.xa_lon)))
    trips = trips.assign(xb=list(zip(trips.xb_lat, trips.xb_lon)))

    f = (40.75798, -73.98550)  # Times Square
    r = 1000  # meters

    trips = trips[trips[['xa', 'xb']].applymap(lambda x: (geodistance(x, f).m <= r)).all(axis=1)]

    return trips


def main():
    for hour in [6, 18]:
        memo = Memo()
        hour = memo(hour)
        query = memo({
            'table_names': {"green_tripdata_2016-05", "yellow_tripdata_2016-05"},
            'sql_where': f"(({hour}) == cast(strftime('%H', [ta]) as int))",
            'sql_limit': 111,
        })
        graph_path = memo(max(DATA.glob(f"*WithLag/*train/**/lag/H={hour}/**/manhattan.pkl")))

        print(memo)

        graph = load_graph(graph_path)
        trips = with_nearest_ingraph(get_trips_times_square(**query), graph)




if __name__ == '__main__':
    main()
