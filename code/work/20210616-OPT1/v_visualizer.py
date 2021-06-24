# RA, 2021-06-18

from twig import log

import logging

from opt_utils.misc import Section

log.parent.handlers[0].setLevel(logging.DEBUG)

import contextlib
from pathlib import Path
from datetime import timedelta, datetime
from more_itertools import pairwise

import numpy as np
import pandas as pd
import networkx as nx

from matplotlib.ticker import MaxNLocator

import json
import percache

from sorcery import unpack_keys as unpack
from inclusive import range

from tcga.utils import unlist1, relpath, mkdir, first, Now, whatsmyname, First
from plox import Plox, rcParam

from opt_maps import maps

from i_infer_trajectories import paths_of_route
from z_sources import get_problem_data, preprocess_problem_data, read_subcase, EDGE_TTT_KEY

BASE = Path(__file__).parent
DATA = next(p for p in BASE.parents for p in p.glob("**/model")).resolve()

cache = percache.Cache(str(mkdir(Path(__file__).with_suffix('')) / f"percache.dat"), livesync=True, repr=repr)
cache.clear(maxage=(60 * 60 * 24))

sql_string = First(str.split).then(' '.join)

style = {rcParam.Font.size: 16, rcParam.Text.usetex: True}

# logging.root.manager.loggerDict.iteritems()
logging.getLogger("PIL").setLevel(logging.WARNING)  # works?


@contextlib.contextmanager
def excess_travel_time_traj(graph, trips: pd.DataFrame, routes: pd.DataFrame, edge_weight=EDGE_TTT_KEY,
                            **kwargs) -> Plox:
    routes = {
        iv: pd.concat(axis=0, objs=list(paths_of_route(route, graph, edge_ttt_weight=edge_weight)))
        for (iv, route) in routes.groupby(by='iv')
    }

    edge_lag = nx.get_edge_attributes(graph, name=edge_weight)

    rng = np.random.default_rng(seed=43)

    with Plox() as px:
        extent = maps.ax4(list(trips.xa_lat) + list(trips.xb_lat), list(trips.xa_lon) + list(trips.xb_lon),
                          extra_space=0.35)

        px.a.imshow(maps.get_map_by_bbox(maps.ax2mb(*extent)), extent=extent, interpolation='quadric', zorder=-100)
        px.a.axis("off")

        px.a.set_xlim(extent[0:2])
        px.a.set_ylim(extent[2:4])

        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap, Normalize

        cmap = LinearSegmentedColormap.from_list('annoyance', ["darkblue", "darkgreen", "darkorange", "darkred"])

        norm = Normalize(vmin=0, vmax=15)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_clim(norm.vmin, norm.vmax)

        from matplotlib.transforms import Bbox
        ((x0, y0), (x1, y1)) = px.a.get_position().get_points()
        cax = px.f.add_axes(Bbox(((x0 + 0.01, y1 - 0.07), (x0 + (x1 - x0) * 0.5, y1 - 0.05))))
        cax.set_title("Excess travel time, min")
        ticks = range[int(norm.vmin), int(norm.vmax) - 1]
        cb = px.f.colorbar(mappable=sm, cax=cax, orientation='horizontal', ticks=ticks, )
        cb.ax.tick_params(labelsize='x-small')

        for (_, trip) in trips.iterrows():
            if not pd.isna(trip.iv):
                route = routes[trip.iv]
                route = route[(trip.iv_ta < route.lag) & (route.lag <= trip.iv_tb)]

                if any(routes[trip.iv].lag.diff().dt.total_seconds() < -2):
                    log.warning(f"route `lag` for route {trip.iv} is far from ordered.")

                # TODO: are some edges missing?
                path = list(pairwise(route.index))
                missing = {e for e in path if e not in edge_lag and (e[0] != e[1])}
                if missing:
                    log.warning(f"missing edges: {missing}")
                    breakpoint()

                short_len = nx.shortest_path_length(graph, trip.ia, trip.ib, weight=edge_weight)
                route_len = sum(((e[0] != e[1]) and edge_lag.get(e, np.nan)) for e in path)
                c = sm.to_rgba((route_len - short_len) / 60)
                s = 10

                (dx, dy) = rng.uniform(0, 0.0005, size=2)
                px.a.plot(route.lon + dx, route.lat + dy, alpha=0.5, lw=0.5, c=c)

                px.a.scatter(trip.xa_lon, trip.xa_lat, zorder=1000, alpha=0.8, s=s, facecolor=c, edgecolor='none')
            else:
                c = "red"
                s = 18
                px.a.scatter(trip.xa_lon, trip.xa_lat, zorder=1000, alpha=0.8, s=s, facecolor='none', edgecolor=c)

        yield px


@contextlib.contextmanager
def visualize3d(graph, trips, routes: pd.DataFrame, **kw):
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
                line=dict(width=0.4, color="green"),
                mode="lines",
                showlegend=False,
            ),
        )

        # Dropoff timewindow
        fig.add_trace(
            go.Scatter3d(
                x=[trip.xb_lon] * 2,
                y=[trip.xb_lat] * 2,
                z=ztime(trip.twb),
                # marker=dict(size=0),
                line=dict(width=0.4, color="red"),
                mode="lines",
                showlegend=False,
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
                    line=dict(width=0.8, color="blue", dash="dash"),
                    mode="lines",
                    showlegend=False,
                ),
            )

    for (n, (iv, route)) in enumerate(routes.groupby('iv')):
        if max(route.load) == 0:
            continue

        from matplotlib.cm import get_cmap as mpl_cmap
        from matplotlib.colors import to_hex
        color = to_hex((mpl_cmap("rainbow"))(n / len(set(routes.iv))))

        # import plotly.express
        # color = plotly.express.colors.qualitative.Dark24[n]

        path = pd.concat(axis=0, objs=list(paths_of_route(route, graph, edge_ttt_weight=EDGE_TTT_KEY)))
        path = path[path.lag <= t_max]

        fig.add_trace(
            go.Scatter3d(
                x=path.lon,
                y=path.lat,
                z=ztime(path.lag),
                # marker=dict(size=0),
                line=dict(width=1, color=color),
                mode="lines",
                name=f"route {iv}",
            ),
        )

    fig.update_layout(
        showlegend=True,
        scene=dict(
            xaxis_title="lon",
            yaxis_title="lat",
            zaxis_title="time, s",
        ),
    )

    yield fig


@contextlib.contextmanager
def excess_travel_time_hist(graph: nx.DiGraph, trips: pd.DataFrame, routes: pd.DataFrame, **kwargs):
    unserviced = np.sum(trips.iv_ta.isna())
    trips = trips[~trips.iv_ta.isna()]
    ref = trips.duration * timedelta(seconds=1)
    old = trips.apply(axis=1, func=(lambda trip: nx.shortest_path_length(graph, trip.ia, trip.ib, weight=EDGE_TTT_KEY)))
    new = (trips.iv_tb - trips.iv_ta).dt.total_seconds()
    with Plox({**style, rcParam.Figure.figsize: (8, 3)}) as px:
        data = (new - old) / 60

        m = 15
        px.a.hist(data, bins=m, range=[0, 15], color="C0", edgecolor="white", lw=2)
        px.a.set_xlabel("Excess travel time, min")
        px.a.set_ylabel("Number of passengers")
        # px.a.set_yticklabels(px.a.get_yticklabels(), fontsize="small")
        px.a.tick_params(axis='y', labelsize="large")
        px.a.grid(True, zorder=-1000, linewidth=0.1)
        px.a.yaxis.set_major_locator(MaxNLocator(integer=True))

        px.a.bar(x=[m + 1], height=[np.sum(data > m)], color="C2")
        px.a.bar(x=[m + 2], height=[unserviced], color="C3")
        px.a.set_xticks(list(range[0, m]) + [m + 1, m + 2])
        px.a.set_xticklabels(list(str(i) for i in range[0, m]) + [f"..."] + ["oo"])
        px.a.tick_params(axis='x', labelsize="large")

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


def plot_all(path_src: Path, path_dst=None, skip_existing=True, do_3d=False):
    path_dst = path_dst or mkdir(path_src / "plots")

    log.info(f"Plotting {relpath(path_src)} -> {relpath(path_dst)}")

    (params, routes, trips) = unpack(read_subcase(path_src))

    # noinspection PyUnresolvedReferences
    from pandas import Timestamp  # required for `eval`
    trips.twa = list(map(eval, trips.twa))
    trips.twb = list(map(eval, trips.twb))

    problem_data = (cache(get_problem_data)(**params['data']))

    if not set(trips.index).issubset(set(problem_data.trips.index)):
        log.warning(f"`problem_data.trips` does not contain the trips from `trips.tsv`")

    if 'data_post' not in params:
        log.warning(f"Imputing `data_post` from the default.")
        from b_casestudent import get_default_params
        params['data_post'] = get_default_params()['data_post']

    problem_data = preprocess_problem_data(problem_data, **params['data_post'])
    problem_data.trips = None

    graph = problem_data.graph

    alles = {'graph': graph, 'trips': trips, 'routes': routes}

    # 3d plot

    if do_3d:
        with Section("Making 3d plot...", out=log.info):
            with visualize3d(**alles) as fig:
                fig.write_html(str(path_dst / f"visualize3d.html"))

    # 2d plots

    ff = [
        excess_travel_time_traj,
        excess_travel_time_hist,
        vehicle_load,
    ]

    for f in ff:
        out_fig = path_dst / f"{f.__name__}.png"
        if skip_existing and out_fig.is_file():
            log.info(f"{relpath(out_fig)} exists, skipping.")
        else:
            with Section(f"Making {relpath(out_fig)}", out=log.info):
                with f(**alles) as px:
                    px.f.savefig(out_fig)


def main():
    path = unlist1(Path(__file__).with_suffix('').glob("sample_data"))

    for subcase in path.glob("*cases/*"):
        assert subcase.is_dir()
        plot_all(path_src=subcase, path_dst=mkdir(path / f"plots/{subcase.name}"), skip_existing=False, do_3d=True)


if __name__ == '__main__':
    main()
