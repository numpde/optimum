# RA, 2021-06-18


EDGE_TTT_KEY = "lag"  # time-to-transition attribute

from twig import log
import logging

log.parent.handlers[0].setLevel(logging.DEBUG)

import contextlib
from pathlib import Path
from datetime import timedelta, datetime
from more_itertools import pairwise

import numpy as np
import pandas as pd
import networkx as nx

import json
import percache

from inclusive import range

from matplotlib.ticker import MaxNLocator

from tcga.utils import unlist1, relpath, mkdir, first, Now, whatsmyname, First
from plox import Plox, rcParam

from opt_maps import maps
from opt_utils.graph import largest_component, GraphNearestNode, GraphPathDist

from z_sources import get_problem_data

BASE = Path(__file__).parent
DATA = next(p for p in BASE.parents for p in p.glob("**/model")).resolve()

cache = percache.Cache(str(mkdir(Path(__file__).with_suffix('')) / f"percache.dat"), livesync=True, repr=repr)
cache.clear(maxage=(60 * 60 * 24))

sql_string = First(str.split).then(' '.join)

style = {rcParam.Font.size: 14}


def paths_of_route(route, graph, edge_weight=EDGE_TTT_KEY):
    node_loc = pd.DataFrame(nx.get_node_attributes(graph, name='loc'), index=["lat", "lon"])
    edge_lag = nx.get_edge_attributes(graph, name=edge_weight)

    for (uv, (ta, tb)) in zip(pairwise(route.i), pairwise(route.est_time_dep)):
        path = nx.shortest_path(graph, *uv, weight=edge_weight)
        path = node_loc[list(path)].T
        path = path.assign(lag=(np.cumsum([ta] + [timedelta(seconds=1) * edge_lag[e] for e in pairwise(path.index)])))

        yield path



@contextlib.contextmanager
def bare_trajectories(graph, trips: pd.DataFrame, routes: pd.DataFrame, edge_weight=EDGE_TTT_KEY, **kwargs) -> Plox:
    # with GraphPathDist(graph, edge_weight=edge_weight) as gpd:
    #     trajectories = list(map(gpd.path_only, zip(trips.ia, trips.ib)))

    nodes = pd.DataFrame(data=nx.get_node_attributes(graph, "loc"), index=["lat", "lon"]).T

    routes = {
        iv: pd.concat(axis=0, objs=list(paths_of_route(route, graph, edge_weight=edge_weight)))
        for (iv, route) in routes.groupby(by='iv')
    }

    rng = np.random.default_rng(seed=43)

    with Plox() as px:
        for (_, trip) in trips.iterrows():
            if not pd.isna(trip.iv):
                route = routes[trip.iv]
                route = route[(trip.iv_ta <= route.lag) & (route.lag <= trip.iv_tb)]
                px.a.plot(route.lon + rng.uniform(0, 1e-3), route.lag + rng.uniform(0, 1e-3), alpha=0.7, lw=0.3)

        #     px.a.plot(*nodes.loc[list(traj), ['lon', 'lat']].values.T, alpha=0.7, lw=0.3)

        # # extent = maps.ax4(nodes.lat, nodes.lon)
        # extent = maps.ax4(list(trips.xa_lat) + list(trips.xb_lat), list(trips.xa_lon) + list(trips.xb_lon))
        #
        # px.a.imshow(maps.get_map_by_bbox(maps.ax2mb(*extent)), extent=extent, interpolation='quadric', zorder=-100)
        # px.a.axis("off")
        #
        # px.a.set_xlim(extent[0:2])
        # px.a.set_ylim(extent[2:4])

        px.show()
        exit()

        yield px


@contextlib.contextmanager
def visualize3d(graph, trips, routes: pd.DataFrame, **kwargs):
    import plotly.graph_objects as go
    import plotly.express as px

    # log.info(f"Solution: \n{trips.sort_values(by=['iv', 'iv_ta']).to_markdown()}")

    rng = np.random.default_rng(seed=43)

    fig = go.Figure()

    t_max = trips[['twa', 'twb']].applymap(np.max).max().max()

    t0 = min(routes.est_time_arr)
    ztime = (lambda v: (pd.Series(v) - t0).dt.total_seconds())

    for (i, trip) in trips.iterrows():
        # color = rng.choice(px.colors.sequential.Plasma)

        # Pickup timewindow
        fig.add_trace(
            go.Scatter3d(
                x=[trip.xa_lon] * 2,
                y=[trip.xa_lat] * 2,
                z=ztime(trip.twa),
                # marker=dict(size=0),
                line=dict(width=0.2, color="green"),
                mode="lines",
            ),
        )

        # Dropoff timewindow
        fig.add_trace(
            go.Scatter3d(
                x=[trip.xb_lon] * 2,
                y=[trip.xb_lat] * 2,
                z=ztime(trip.twb),
                # marker=dict(size=0),
                line=dict(width=0.2, color="red"),
                mode="lines",
            ),
        )

        # Connect
        if not pd.isna(trip.iv_ta) and not pd.isna(trip.iv_ta):
            fig.add_trace(
                go.Scatter3d(
                    x=[trip.xa_lon, trip.xb_lon],
                    y=[trip.xa_lat, trip.xb_lat],
                    z=ztime([trip.iv_ta, trip.iv_tb]),
                    # marker=dict(size=0),
                    line=dict(width=0.8, color="green", dash="dash"),
                    mode="lines",
                ),
            )

    for (iv, route) in routes.groupby('iv'):
        if max(route.load) == 0:
            continue

        color = rng.choice(["black", "blue", "brown", "magenta"])

        # log.debug(f"Route: \n{route.to_markdown()}")

        path = pd.concat(axis=0, objs=paths_of_route(route, graph, edge_weight=EDGE_TTT_KEY))

        path = path[path.lag <= t_max]

        fig.add_trace(
            go.Scatter3d(
                x=path.lon,
                y=path.lat,
                z=ztime(path.lag),
                # marker=dict(size=0),
                line=dict(width=0.6, color=color),
                mode="lines",
            ),
        )

    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis_title="lon",
            yaxis_title="lat",
            zaxis_title="time, s",
        ),
    )

    yield fig


@contextlib.contextmanager
def excess_trip_durations(graph: nx.DiGraph, trips: pd.DataFrame, routes: pd.DataFrame, **kwargs):
    unserviced = np.sum(trips.iv_ta.isna())
    trips = trips[~trips.iv_ta.isna()]
    ref = trips.duration * timedelta(seconds=1)
    old = trips.apply(axis=1, func=(lambda trip: nx.shortest_path_length(graph, trip.ia, trip.ib, weight=EDGE_TTT_KEY)))
    new = (trips.iv_tb - trips.iv_ta).dt.total_seconds()
    with Plox({**style, rcParam.Figure.figsize: (8, 3)}) as px:
        data = (new - old) / 60

        # DEBUG
        # data = np.arange(0, 20)

        m = 15
        px.a.hist(data, bins=m, range=[0, 15], color="C0")
        px.a.set_xlabel("Excess trip duration, min")
        px.a.set_ylabel("Number of passengers")
        # px.a.set_yticklabels(px.a.get_yticklabels(), fontsize="small")
        px.a.tick_params(axis='y', labelsize="x-small")
        px.a.grid(True, zorder=-1000, linewidth=0.1)
        px.a.yaxis.set_major_locator(MaxNLocator(integer=True))

        px.a.bar(x=[m + 1], height=[np.sum(data > m)], color="C2")
        px.a.bar(x=[m + 2], height=[unserviced], color="C3")
        px.a.set_xticks(list(range[0, m]) + [m + 1, m + 2])
        px.a.set_xticklabels(list(str(i) for i in range[0, m]) + [f"..."] + ["oo"])
        px.a.tick_params(axis='x', labelsize="x-small")

        # px.a.set_yscale('log')
        # px.a.set_yticks(range(int(max(px.a.get_yticks()))))

        yield px


@contextlib.contextmanager
def vehicle_load(routes: pd.DataFrame, **kwargs):
    rng = np.random.default_rng(seed=43)

    with Plox({**style, rcParam.Figure.figsize: (8, 3)}) as px:
        for (iv, route) in routes.groupby(by='iv'):
            if any(route.load):
                route = route.reset_index()
                # route = route.loc[(min(route[route.load != 0].index) - 1):]
                tb = max(route.est_time_arr[route.load != 0])
                route = route[route.est_time_arr <= tb + timedelta(minutes=5)]
                time = route.est_time_arr
                route.load += 0.02 * (route.load != 0) * rng.uniform(-1, 1, size=len(route.load))
                px.a.step((time - min(time)).dt.total_seconds() / 60, route.load, '-', where='post', lw=2)

        px.a.yaxis.set_major_locator(MaxNLocator(integer=True))

        px.a.set_xlabel("Time offset, min")
        px.a.set_ylabel("Vehicle load")

        yield px


def plot_all(path_src: Path, path_dst=None, skip_existing=True):
    path_dst = path_dst or mkdir(path_src / "plots")

    log.info(f"Plotting {relpath(path_src)} -> {relpath(path_dst)}")

    with unlist1(path_src.glob("params.json")).open(mode='r') as fd:
        params = json.load(fd)

    with unlist1(path_src.glob("routes.tsv")).open(mode='r') as fd:
        routes = pd.read_table(fd, parse_dates=['est_time_arr', 'est_time_dep'])

    with unlist1(path_src.glob("trips.tsv")).open(mode='r') as fd:
        trips = pd.read_table(fd, parse_dates=['ta', 'tb', 'iv_ta', 'iv_tb'])

    # noinspection PyUnresolvedReferences
    from pandas import Timestamp  # required for `eval`
    trips.twa = list(map(eval, trips.twa))
    trips.twb = list(map(eval, trips.twb))

    graph = (cache(get_problem_data)(**params['data'])).graph

    alles = {'graph': graph, 'trips': trips, 'routes': routes}

    ff = [
        bare_trajectories,
        excess_trip_durations,
        vehicle_load,
    ]

    for f in ff:
        out_fig = path_dst / f"{f.__name__}.png"
        if skip_existing and out_fig.is_file():
            log.info(f"{relpath(out_fig)} exists, skipping.")
        else:
            log.info(f"Making {relpath(out_fig)}...")
            with f(**alles) as px:
                px.f.savefig(out_fig)

    # with visualize3d(**alles) as fig:
    #     fig.write_html(str(out_dir / f"visualize3d.html"))


def main():
    path = unlist1(Path(__file__).with_suffix('').glob("sample_data"))

    for subcase in path.glob("*cases/*"):
        plot_all(path_src=subcase, path_dst=mkdir(path / f"plots/{subcase.name}"), skip_existing=False)


if __name__ == '__main__':
    main()
