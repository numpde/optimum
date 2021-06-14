# RA, 2019-10-27
# RA, 2021-06-12

"""
Based on
https://github.com/numpde/transport/blob/master/pt2pt/20191021-NYCTLC/data_preparation/f_explore_taxidata2.py
"""

import contextlib
import logging

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

from opt_maps import maps
from opt_utils.graph import largest_component, GraphNearestNode, GraphPathDist
from opt_utils.style import default_style, name2color, name2cmap, get_velocity_cmap
from opt_utils.misc import Section

BASE = Path(__file__).parent
DATA = next(p for p in BASE.parents for p in p.glob("**/data")).resolve()

out_dir = mkdir(Path(__file__).with_suffix(''))

parallel_map = map

cache = percache.Cache(str(mkdir(Path(__file__).parent / "cache") / "percache.dat"), livesync=True)
cache.clear(maxage=(60 * 60 * 24 * 7))


# ==  DATA  == #

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
    file = unlist1(BASE.glob(f"**/*graph/{area}.pkl"))

    with file.open(mode='rb') as fd:
        import pickle
        g = pickle.load(fd)

    assert (type(g) is nx.DiGraph), \
        f"{relpath(file)} has wrong type {type(g)}."

    return g


def get_trips(table_name, where="", order="random()", limit=11111) -> pd.DataFrame:
    with Section("Querying trips", out=log.debug):
        sql = " ".join([
            f"SELECT * FROM [{table_name}]",
            f"WHERE    ({where}) " if where else "",
            f"ORDER BY ({order}) " if order else "",
            f"LIMIT    ({limit}) " if limit else "",
        ])

        return query_trips(sql)


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


def with_shortest_distance(trips, graph):
    with Section("Computing shortest distances", out=log.debug):
        return trips.join(
            pd.DataFrame(
                data=parallel_map(GraphPathDist(graph, edge_weight="len"), zip(trips.ia, trips.ib)),
                index=trips.index,
                columns=['path', 'shortest'],
            )
        )


@cache
def get_trips_mit_alles(area: str, table_name: str):
    graph = largest_component(get_road_graph(area))

    trips = get_trips(table_name)
    trips = with_nearest_ingraph(trips, graph)
    trips = with_shortest_distance(trips, graph)

    return pd.Series({'area': area, 'table_name': table_name, 'graph': graph, 'trips': trips})


# ==  PLOTS  == #

@contextlib.contextmanager
def trip_distance_vs_shortest(data) -> Plox:
    trips = data.trips

    # On-graph distance vs reported distance [meters]
    df = pd.DataFrame(trips.shortest).assign(reported=trips.distance)

    KM = 10  # km

    # Convert to [km] and stay below KM
    df = df.applymap(lambda x: (x / 1e3))
    df = df.applymap(lambda km: (km if (km < KM) else np.nan)).dropna()

    # Hour of the day
    df['h'] = trips['ta'].dt.hour

    import matplotlib.pyplot as plt

    with Plox(default_style) as px:
        px.a.set_aspect(aspect="equal", adjustable="box")
        px.a.grid()
        px.a.plot(*(2 * [[0, df[['reported', 'shortest']].values.max()]]), c='k', ls='--', lw=0.3, zorder=100)
        for (h, hdf) in df.groupby(df['h']):
            c = plt.get_cmap("twilight_shifted")([h / 24])
            px.a.scatter(
                hdf.reported, hdf.shortest,
                c=c, s=3, alpha=0.7, lw=0, zorder=10,
                label=(f"{len(hdf)} trips at {h}h")
            )
        px.a.set_xlabel("Reported distance, km")
        px.a.set_ylabel("Graph distance, km")
        px.a.set_xticks(range(KM + 1))
        px.a.set_yticks(range(KM + 1))
        px.a.legend(fontsize=5)

        yield px


@contextlib.contextmanager
def trip_trajectories_ingraph(data) -> Plox:
    table_name: str = data.table_name
    trips: pd.DataFrame = data.trips
    graph: nx.DiGraph = data.graph

    # Max number of trajectories to plot
    N = 1000

    nodes = pd.DataFrame(data=nx.get_node_attributes(graph, "loc"), index=["lat", "lon"]).T

    with Section("Subsampling trips", out=log.debug):
        trips = trips.sample(min(N, len(trips)), replace=False, random_state=43)
        log.debug(f"{len(trips)} trips")

    with Section("Computing trajectories", out=log.debug):
        trajectories = parallel_map(GraphPathDist(graph).path_only, zip(trips.ia, trips.ib))

    with Section("Getting the background OSM map", out=log.debug):
        extent = maps.ax4(nodes.lat, nodes.lon)
        background = maps.get_map_by_bbox(maps.ax2mb(*extent))

    with Plox({**default_style, 'font.size': 5}) as px:
        px.a.imshow(background, extent=extent, interpolation='quadric', zorder=-100)

        px.a.axis("off")

        px.a.set_xlim(extent[0:2])
        px.a.set_ylim(extent[2:4])

        for traj in trajectories:
            (lat, lon) = nodes.loc[list(traj), ['lat', 'lon']].values.T
            px.a.plot(lon, lat, c=name2color(table_name), alpha=0.1, lw=0.3)

        yield px


@contextlib.contextmanager
def trip_trajectories_velocity(data) -> Plox:
    trips: pd.DataFrame = data.trips
    graph: nx.DiGraph = data.graph

    H = [7, 8]  # Hours of the day
    N = 10000  # Max number of trajectories to use

    nodes = pd.DataFrame(data=nx.get_node_attributes(graph, "loc"), index=["lat", "lon"]).T

    with Section("Subsampling trips", out=log.debug):
        trips = trips[trips['ta'].dt.hour.isin(H)]
        trips = trips.sample(min(N, len(trips)), replace=False, random_state=43)

        trips['velocity'] = trips['distance'] / trips['duration']
        trips = trips.sort_values(by='velocity', ascending=True)

        log.debug(f"{len(trips)} trips.")

    with Section("Computing estimated trajectories", out=log.debug):
        trips['traj'] = list(parallel_map(GraphPathDist(graph).path_only, zip(trips.ia, trips.ib)))

    with Section("Getting the background OSM map", out=log.debug):
        extent = maps.ax4(nodes.lat, nodes.lon)
        background = maps.get_map_by_bbox(maps.ax2mb(*extent))

    with Section("Computing edge velocities", out=log.debug):
        edge_vel = pd.DataFrame(
            data=[(e, v) for (traj, v) in zip(trips.traj, trips.velocity) for e in pairwise(traj)],
            columns=['edge', 'velocity'],
        ).groupby('edge').mean().velocity

        assert len(edge_vel)

    with Plox({**default_style, 'font.size': 5}) as px:
        px.a.imshow(background, extent=extent, interpolation='quadric', zorder=-100)

        px.a.axis("off")

        px.a.set_xlim(extent[0:2])
        px.a.set_ylim(extent[2:4])

        edge_vel: pd.Series = edge_vel.clip(lower=1, upper=7)

        nx.draw_networkx_edges(
            graph.edge_subgraph(edge_vel.index),
            ax=px.a,
            pos=nx.get_node_attributes(graph, name="pos"),
            edgelist=edge_vel.index,
            edge_color=edge_vel,
            edge_cmap=get_velocity_cmap(),
            arrows=False, node_size=0, alpha=0.8, width=0.3,
        )

        import matplotlib.cm
        import matplotlib.colors
        norm = matplotlib.colors.Normalize(vmin=edge_vel.min(), vmax=edge_vel.max())
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=get_velocity_cmap())

        from matplotlib.transforms import Bbox
        ((x0, y0), (x1, y1)) = px.a.get_position().get_points()
        cax: plt.Axes = px.f.add_axes(Bbox(((x0 + 0.01, y1 - 0.07), (x0 + (x1 - x0) * 0.5, y1 - 0.05))))
        cax.set_title("Est. velocity, m/s")
        ticks = range[int(edge_vel.min() + 1), int(edge_vel.max() - 1)]
        px.f.colorbar(mappable=sm, cax=cax, orientation='horizontal', ticks=ticks)

        yield px


@contextlib.contextmanager
def compare_multiple_trajectories(data) -> Plox:
    graph = data.graph
    trips = data.trips

    # Number of trips to plot (max)
    N = 9
    # The trips should be this long
    min_distance = 2000
    max_distance = 4000
    # Number of trajectories per trip
    M = 17
    # Hours of the day
    H = [7, 8]

    nodes = pd.DataFrame(data=nx.get_node_attributes(graph, "loc"), index=["lat", "lon"]).T
    edges_len = nx.get_edge_attributes(graph, name="len")

    with Section("Subsampling trips", out=log.debug):
        trips = trips[trips['ta'].dt.hour.isin(H)]
        trips = trips[(min_distance <= trips.distance) & (trips.distance <= max_distance)]
        trips = trips.sample(min(N, len(trips)), replace=False, random_state=43)
        log.debug(f"{len(trips)} trips.")

    with Section("Getting the background OSM map", out=log.debug):
        extent = maps.ax4(nodes.lat, nodes.lon)
        background = maps.get_map_by_bbox(maps.ax2mb(*extent))

    with Plox({**default_style, 'font.size': 5}) as px:
        px.a.imshow(background, extent=extent, interpolation='quadric', zorder=-100)

        px.a.axis("off")

        px.a.set_xlim(extent[0:2])
        px.a.set_ylim(extent[2:4])

        for (n, (__, trip)) in enumerate(trips.iterrows(), start=1):
            with Section("Computing candidate trajectories", out=log.debug):
                trajectories = pd.DataFrame(data={'path': [
                    path
                    for (__, path) in
                    zip(range(M), nx.shortest_simple_paths(graph, source=trip.ia, target=trip.ib))
                ]})
                trajectories['dist'] = [sum(edges_len[e] for e in pairwise(path)) for path in trajectories.path]
                trajectories = trajectories.sort_values(by='dist', ascending=False)

            # marker = dict(markersize=2, markeredgewidth=0.2, markerfacecolor="None")
            # px.a.plot(trip.xa_lon, trip.xa_lat, 'og', **marker)
            # px.a.plot(trip.xb_lon, trip.xb_lat, 'xr', **marker)

            px.a.text(trip.xa_lon, trip.xa_lat, s=f"{n}", fontdict={'size': 6}, color='g')
            px.a.text(trip.xb_lon, trip.xb_lat, s=f"{n}", fontdict={'size': 6}, color='r')

            cmap = get_velocity_cmap()
            colors = cmap(pd.Series(trajectories['dist'] / trip['distance']).rank(pct=True))

            for (c, path) in zip(colors, trajectories.path):
                (lat, lon) = nodes.loc[list(path), ['lat', 'lon']].values.T
                px.a.plot(lon, lat, c=c, alpha=0.1, lw=0.3)

        yield px


# ==  MAIN  == #

def test():
    with Section("Test: query", out=log.debug):
        log.debug(f'\n{query_trips("SELECT * FROM [green_tripdata_2016-05] LIMIT 2").to_markdown()}')


def main(area):
    tables = {"green_tripdata_2016-05", "yellow_tripdata_2016-05"}

    for table_name in sorted(tables):
        data = get_trips_mit_alles(area, table_name)

        ff = [
            trip_trajectories_velocity,
            compare_multiple_trajectories,
            trip_trajectories_ingraph,
            trip_distance_vs_shortest,
        ]

        for f in ff:
            with Section(f"{f.__name__}", out=log.debug):
                with f(data) as px:
                    file = mkdir(out_dir / f.__name__) / f"{table_name}.png"
                    px.f.savefig(file)
                    log.debug(f"Written: {relpath(file)}")


if __name__ == '__main__':
    test()
    main("manhattan")
