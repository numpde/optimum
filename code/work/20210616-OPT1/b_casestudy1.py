# RA, 2021-06-15

"""

Focus on a short time-window around Times Square.

==

Notes:

On wait time
https://github.com/google/or-tools/issues/765

Working example
https://github.com/google/or-tools/blob/stable/ortools/constraint_solver/samples/cvrptw.py
"""

import ortools

from more_itertools import pairwise

EDGE_TTT_KEY = "lag"  # time-to-transition attribute

import tensorflow as tf

import contextlib

from twig import log
import logging

log.parent.handlers[0].setLevel(logging.DEBUG)

from typing import List
from pathlib import Path
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import networkx as nx

import percache

from geopy.distance import distance as geodistance

from tcga.utils import unlist1, relpath, mkdir, first, Now, whatsmyname
from plox import Plox

from opt_maps import maps
from opt_utils.graph import largest_component, GraphNearestNode, GraphPathDist
from opt_utils.misc import Section
from opt_trips.trips import get_raw_trips, KEEP_COLS, with_nearest_ingraph, with_shortest_distance

BASE = Path(__file__).parent
DATA = next(p for p in BASE.parents for p in p.glob("**/model")).resolve()
graph_source = max(DATA.glob(f"*WithLag/*train/**/lag/H=18"))

out_dir = mkdir(Path(__file__).with_suffix(''))

AREA = "manhattan"
table_names = sorted({"green_tripdata_2016-05", "yellow_tripdata_2016-05"})

# Focal point
TIMES_SQUARE = (40.75798, -73.98550)
TIME_WINDOW = 15  # minutes


def myrepr(obj):
    return repr(obj)


cache = percache.Cache(str(out_dir / f"percache.dat"), livesync=True, repr=myrepr)
cache.clear(maxage=(60 * 60 * 24 * 7))


# https://developers.google.com/optimization/routing/routing_options
class OrStatus:
    NOT_YET = 0
    SUCCESS = 1
    FAIL_NOT_FOUND = 2
    FAIL_TIMEOUT = 3
    INVALID = 4


def load_graph(area):
    file = max(graph_source.glob(f"**/{area}.pkl"))
    log.debug(f"Graph file: {relpath(file)}")

    with file.open(mode='rb') as fd:
        import pickle
        g = pickle.load(fd)

    assert (type(g) is nx.DiGraph), \
        f"{relpath(file)} has wrong type {type(g)}."

    assert nx.get_edge_attributes(g, name=EDGE_TTT_KEY)

    return g


@contextlib.contextmanager
def plot_trajectories(graph, trips, edge_weight=EDGE_TTT_KEY) -> Plox:
    with GraphPathDist(graph, edge_weight=edge_weight) as gpd:
        trajectories = list(map(gpd.path_only, zip(trips.ia, trips.ib)))

    nodes = pd.DataFrame(data=nx.get_node_attributes(graph, "loc"), index=["lat", "lon"]).T

    with Plox() as px:
        extent = maps.ax4(nodes.lat, nodes.lon)

        px.a.imshow(maps.get_map_by_bbox(maps.ax2mb(*extent)), extent=extent, interpolation='quadric', zorder=-100)
        px.a.axis("off")

        px.a.set_xlim(extent[0:2])
        px.a.set_ylim(extent[2:4])

        for traj in trajectories:
            px.a.plot(*nodes.loc[list(traj), ['lon', 'lat']].values.T, alpha=0.7, lw=0.3)

        yield px


def attach_timewindows(trips: pd.DataFrame):
    (a_early, a_late) = (timedelta(minutes=2), timedelta(minutes=10))
    (b_early, b_late) = (timedelta(minutes=20), timedelta(minutes=10))
    trips = trips.assign(twa=trips.ta.apply(lambda t: (t - a_early, t + a_late)))
    trips = trips.assign(twb=trips.tb.apply(lambda t: (t - b_early, t + b_late)))
    return trips


def concentrated_subset(trips: pd.DataFrame):
    max_dist = 2000  # meters

    dist_to_times_square = pd.Series(index=trips.index, data=[
        max(geodistance(row.xa, TIMES_SQUARE).m, geodistance(row.xb, TIMES_SQUARE).m)
        for (i, row) in trips.iterrows()
    ])

    trips = trips[dist_to_times_square <= max_dist]
    return trips


@cache
def reduce_to_clique(problem: pd.Series, hash=None):
    log.debug(f"{whatsmyname()} is busy now...")

    # 0. Identify a depot node

    n_depot = unlist1(GraphNearestNode(problem.graph)([TIMES_SQUARE]).index)
    assert all(np.isclose(TIMES_SQUARE, problem.graph.nodes[n_depot]['loc'], rtol=1e-3))

    # 1. Make clique graph of nodes-of-interest

    # Put `depot` first
    support_nodes = [n_depot] + sorted(set(problem.trips.ia) | set(problem.trips.ib))

    # Note: This may be unnecessary (and somewhat restrictive)
    assert pd.Series(support_nodes).is_unique

    graph = nx.from_dict_of_dicts(create_using=nx.DiGraph, d={
        # all-to-all distance matrix
        a: {
            b: {EDGE_TTT_KEY: nx.shortest_path_length(problem.graph, a, b, weight=EDGE_TTT_KEY)}
            for b in support_nodes
            if (a != b)
        }
        for a in support_nodes
    })

    # 2. Convert node names to consecutive integers

    parent_attr = 'parent'

    graph = nx.convert_node_labels_to_integers(graph, label_attribute=parent_attr)
    trips = pd.DataFrame(problem.trips)

    # node #0 is the depot
    assert (graph.nodes[0][parent_attr] == n_depot)

    n2i = {o: i for (i, o) in nx.get_node_attributes(graph, name=parent_attr).items()}

    trips.ia = trips.ia.map(n2i)
    trips.ib = trips.ib.map(n2i)

    # 3. Shadow nodes to be serviced

    # Nodes are consecutive integers now
    assert set(graph.nodes) == set(range(len(graph.nodes)))

    def shadow(i):
        assert i in graph.nodes

        # New node
        j = len(graph.nodes)

        graph.add_node(j, **graph.nodes[i])

        for (a, _, d) in graph.in_edges(i, data=True):
            graph.add_edge(a, i, **d)

        for (_, b, d) in graph.out_edges(i, data=True):
            graph.add_edge(i, b, **d)

        shadow_eps = 1  # seconds

        # Connect shadow node to parent
        graph.add_edge(i, j, **{'len': 0, EDGE_TTT_KEY: shadow_eps})
        graph.add_edge(j, i, **{'len': 0, EDGE_TTT_KEY: shadow_eps})

        return j

    trips = trips.assign(ia=trips.ia.apply(shadow))
    trips = trips.assign(ib=trips.ib.apply(shadow))

    problem = problem.copy()
    problem['graph'] = graph
    problem['trips'] = trips
    problem['original_node'] = nx.get_node_attributes(graph, name=parent_attr)

    return problem


@cache
def get_problem_data(area: str, where: str, max_trips: int):
    trips = pd.concat(axis=0, objs=[
        get_raw_trips(table_name, where=where).assign(table_name=table_name)
        for table_name in sorted(table_names)
    ])

    assert len(trips), \
        f"Query returned zero trips. Maybe a misshappen `where`: \n{where}"

    trips = trips[list(KEEP_COLS)].sort_values(by='ta')
    trips = trips.assign(xa=list(zip(trips.xa_lat, trips.xa_lon)))
    trips = trips.assign(xb=list(zip(trips.xb_lat, trips.xb_lon)))

    graph = largest_component(load_graph(area))
    trips = with_nearest_ingraph(trips, graph)

    trips = attach_timewindows(trips)

    trips = trips.head(max_trips)

    log.debug(f"Trips: \n{trips.head(3).to_markdown()} \netc.")
    log.info(f"Number of trips: {len(trips)}.")

    # Simplify problem (subset in space-time)
    trips = concentrated_subset(trips)

    with plot_trajectories(graph, trips) as px:
        px.f.savefig(out_dir / f"bare_trajectories.png")

    return pd.Series({'area': area, 'table_names': table_names, 'graph': graph, 'trips': trips})


def solve(problem: pd.Series, **params):
    graph: nx.DiGraph = problem.graph
    trips: pd.DataFrame = problem.trips

    num_vehicles = params['num_vehicles']  # Number of buses
    cap_vehicles = params['cap_vehicles']  # Bus passenger capacity

    depot = 0
    capacities = [cap_vehicles] * num_vehicles

    assert (graph.number_of_nodes() <= 1000), \
        f"The graph seems too large for this code."

    assert (depot in graph.nodes), \
        f"The depot node {depot} not found in the graph."

    assert (depot not in (set(trips.ia) | set(trips.ib))), \
        f"The depot node should not be a trip endpoint."

    assert nx.is_strongly_connected(graph), \
        f"All graph nodes should be reachable from any other."

    assert 'n' in trips.columns, \
        f"Number of passengers 'n' not given."

    time_mat = pd.DataFrame(nx.floyd_warshall_numpy(graph, weight=EDGE_TTT_KEY), index=graph.nodes, columns=graph.nodes)

    # Distance should be integers (ortools)
    # https://developers.google.com/optimization/routing/tsp
    time_mat = time_mat.round().astype(int)

    # Max time in seconds to reach any point on graph from the depot
    time_radius = max(time_mat[depot])

    # Beginning of time
    t0 = min(set(trips.ta)) - 2 * timedelta(seconds=time_radius)
    rel_t = (lambda t: int((t - t0).total_seconds()))

    # ORTOOLS

    from ortools.constraint_solver.pywrapcp import RoutingIndexManager, RoutingModel
    from ortools.constraint_solver.pywrapcp import DefaultRoutingSearchParameters
    from ortools.constraint_solver.pywrapcp import Assignment
    from ortools.constraint_solver.routing_enums_pb2 import FirstSolutionStrategy

    manager = RoutingIndexManager(graph.number_of_nodes(), num_vehicles, depot)
    routing = RoutingModel(manager)

    # Travel time

    def time_callback(ja, jb):
        return time_mat.loc[manager.IndexToNode(ja), manager.IndexToNode(jb)]
        # return nx.shortest_path_length(graph, manager.IndexToNode(ja), manager.IndexToNode(jb), weight=EDGE_TTT_KEY)

    ix_transit_time = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(ix_transit_time)

    routing.AddDimension(
        evaluator_index=ix_transit_time,
        name='travel_time',
        # allow waiting time:
        slack_max=(60 * 10),
        # vehicle maximum travel time:
        capacity=(60 * 60 * 24),
        # the start time is a window specified below
        fix_start_cumul_to_zero=False,
    )

    dim_travel_time = routing.GetDimensionOrDie('travel_time')
    dim_travel_time.SetGlobalSpanCostCoefficient(100)

    # Register all slack variables with the `assignment` -- ?

    # for i in graph.nodes:
    #     routing.AddToAssignment(dim_travel_time.SlackVar(manager.NodeToIndex(i)))

    # Vehicle start time window

    for iv in range(num_vehicles):
        dim_travel_time.CumulVar(routing.Start(iv)).SetRange(rel_t(t0), rel_t(t0 + timedelta(hours=3)))
        routing.AddToAssignment(dim_travel_time.SlackVar(routing.Start(iv)))

    for iv in range(num_vehicles):
        routing.AddVariableMinimizedByFinalizer(dim_travel_time.CumulVar(routing.Start(iv)))
        routing.AddVariableMinimizedByFinalizer(dim_travel_time.CumulVar(routing.End(iv)))

    # Pickup & delivery locations

    for (_, trip) in trips.iterrows():
        (ja, jb) = map(manager.NodeToIndex, (trip.ia, trip.ib))
        routing.AddPickupAndDelivery(ja, jb)
        routing.solver().Add(routing.VehicleVar(ja) == routing.VehicleVar(jb))
        routing.solver().Add(dim_travel_time.CumulVar(ja) <= dim_travel_time.CumulVar(jb))
        # Note: CumulVar is time of arrival

    # Pickup & delivery time-windows

    for (_, trip) in trips.iterrows():
        (ja, jb) = map(manager.NodeToIndex, (trip.ia, trip.ib))
        dim_travel_time.CumulVar(ja).SetRange(*map(rel_t, trip.twa))  # pickup
        dim_travel_time.CumulVar(jb).SetRange(*map(rel_t, trip.twb))  # delivery
        routing.AddToAssignment(dim_travel_time.SlackVar(ja))
        routing.AddToAssignment(dim_travel_time.SlackVar(jb))

    # Demands & capacities
    # https://developers.google.com/optimization/routing/penalties

    demands = (
            trips[['ia', 'n']].set_index('ia').n.reindex(graph.nodes).fillna(0).astype(int) -
            trips[['ib', 'n']].set_index('ib').n.reindex(graph.nodes).fillna(0).astype(int)
    )

    log.debug(f"Demands: {demands.to_dict()}.")

    def demand_callback(j):
        return demands[manager.IndexToNode(j)]

    ix_transit_demand = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        evaluator_index=ix_transit_demand,
        vehicle_capacities=capacities,
        name='vehicle_capacity',
        # there is no capacity slack:
        slack_max=0,
        # initially, the vehicles are empty:
        fix_start_cumul_to_zero=True,
    )

    dim_capacity = routing.GetDimensionOrDie('vehicle_capacity')
    dim_capacity.SetGlobalSpanCostCoefficient(0)

    # Allow to drop nodes

    penalty_unserviced = 10000  # TODO:?

    for node in demands[demands != 0].index:
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty_unserviced)

    # Initialize solver parameters

    params = DefaultRoutingSearchParameters()
    params.first_solution_strategy = FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.time_limit.seconds = 60 * 30  # TODO
    # params.solution_limit = 100  # TODO

    # Solve

    with Section("Running `ortools` solver", out=log.info):
        assignment: Assignment = routing.SolveWithParameters(params)

    log.info(f"Assignment: {str(assignment)[0:40]}...")
    log.info(f"Success: {routing.status() == OrStatus.SUCCESS}")

    # Repackage the solution

    assignment.ObjectiveValue()

    def unserviced_nodes(assignment):
        for j in range(routing.Size()):
            if not (routing.IsStart(j) or routing.IsEnd(j)):
                if (j == assignment.Value(routing.NextVar(j))):
                    yield manager.IndexToNode(j)

    unserviced = list(unserviced_nodes(assignment))

    log.info(f"Unserviced nodes: {unserviced}.")

    def get_routes():
        for iv in range(manager.GetNumberOfVehicles()):
            def get_route_from(j):
                while True:
                    i = manager.IndexToNode(j)
                    slack = assignment.Value(dim_travel_time.SlackVar(j)) if routing.IsStart(j) else np.nan
                    yield (i, j, assignment.Value(dim_capacity.CumulVar(j)), slack)
                    if routing.IsEnd(j):
                        break
                    else:
                        j = assignment.Value(routing.NextVar(j))

            route = pd.DataFrame(data=get_route_from(routing.Start(iv)), columns=['i', 'j', 'load', 'slack'])

            route = route.assign(cost=np.cumsum(
                [0] +
                [routing.GetArcCostForVehicle(*e, iv) for e in pairwise(route.i)]
            ))

            yield route

    routes = list(get_routes())

    # [assignment.Value(dim_travel_time.SlackVar(i)) for i in demands[demands != 0].index]
    # [assignment.Value(dim_travel_time.SlackVar(manager.NodeToIndex(i))) for i in trips.ia]

    # Sanity check
    for (iv, route) in enumerate(routes):
        assert (routing.Start(iv) == first(route.j))

    # Time of arrival at each intermediate node of route, according to the solver

    routes = [
        route.assign(est_time_arr=[
            t0 + timedelta(seconds=1) * assignment.Value(dim_travel_time.CumulVar(j))
            for j in route.j
        ])
        for route in routes
    ]

    # Estimate the time of departure

    routes = [
        route.assign(est_time_dep=(
                [
                    tb - timedelta(seconds=1) * nx.shortest_path_length(graph, a, b, weight=EDGE_TTT_KEY)
                    for ((a, b), (_, tb)) in zip(pairwise(route.i), pairwise(route.est_time_arr))
                ] + [np.nan]
        ))
        for route in routes
    ]

    # Infer pickups and deliveries

    trips = trips.assign(iv=np.nan, iv_ta=np.nan, iv_tb=np.nan)

    for (i, trip) in trips.iterrows():
        if trip.ia not in unserviced:
            for (iv, route) in enumerate(routes):
                if trip.ia in set(route.i):
                    route = route.loc[unlist1(route[route.i == trip.ia].index):]
                    log.debug(f"\n{route.head(3)}")
                    trips.loc[i, 'iv'] = iv
                    trips.loc[i, 'iv_ta'] = first(route.est_time_dep)
                    trips.loc[i, 'iv_tb'] = route.est_time_arr[unlist1(route[route.i == trip.ib].index)]

    # log.info(f"Solution: \n{trips.sort_values(by=['iv', 'iv_ta']).to_markdown()}")

    return (trips, routes)


def compute_all(**params):
    problem_data_params = dict(area=params['area'], where=params['where'], max_trips=params['problem_data_max_trips'])
    problem_data = get_problem_data(**problem_data_params)

    reduced_problem_data = reduce_to_clique(problem_data, hash=problem_data_params)
    (trips, routes) = solve(reduced_problem_data, **params)

    trips = trips.assign(ia=trips.ia.map(reduced_problem_data.original_node))
    trips = trips.assign(ib=trips.ib.map(reduced_problem_data.original_node))

    routes = [
        route.assign(i=route.i.map(reduced_problem_data.original_node))
        for route in routes
    ]

    log.info(f"Solution: \n{trips.sort_values(by=['iv', 'iv_ta']).to_markdown()}")

    return (problem_data.graph, trips, routes)


def visualize(graph, trips, routes):
    import plotly.graph_objects as go
    import plotly.express as px

    # log.info(f"Solution: \n{trips.sort_values(by=['iv', 'iv_ta']).to_markdown()}")

    rng = np.random.default_rng(seed=43)

    node_loc = pd.DataFrame(nx.get_node_attributes(graph, name='loc'), index=["lat", "lon"])
    edge_lag = nx.get_edge_attributes(graph, name=EDGE_TTT_KEY)

    fig = go.Figure()

    for (i, trip) in trips.iterrows():
        # color = rng.choice(px.colors.sequential.Plasma)

        # Pickup timewindow
        fig.add_trace(
            go.Scatter3d(
                x=[trip.xa_lon] * 2,
                y=[trip.xa_lat] * 2,
                z=trip.twa,
                # marker=dict(size=0),
                line=dict(width=2, color="green"),
                mode="lines",
            ),
        )

        # Dropoff timewindow
        fig.add_trace(
            go.Scatter3d(
                x=[trip.xb_lon] * 2,
                y=[trip.xb_lat] * 2,
                z=trip.twb,
                # marker=dict(size=0),
                line=dict(width=2, color="red"),
                mode="lines",
            ),
        )

        # Connect
        if not pd.isna(trip.iv_ta) and not pd.isna(trip.iv_ta):
            fig.add_trace(
                go.Scatter3d(
                    x=[trip.xa_lon, trip.xb_lon],
                    y=[trip.xa_lat, trip.xb_lat],
                    z=[trip.iv_ta, trip.iv_tb],
                    # marker=dict(size=0),
                    line=dict(width=1, color="green", dash="dash"),
                    mode="lines",
                ),
            )

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=path.lon,
        #         y=path.lat,
        #         # z=path.lag,
        #         z=(path.lag - t0).dt.total_seconds() / 60,
        #         # marker=dict(size=0),
        #         line=dict(width=1, color=color),
        #         mode="lines",
        #     ),
        # )

    for route in routes:
        if max(route.load) == 0:
            continue

        color = rng.choice(["black", "blue", "brown", "magenta"])

        log.debug(f"Route: \n{route.to_markdown()}")

        for (uv, (ta, tb)) in zip(pairwise(route.i), pairwise(route.est_time_dep)):
            path = nx.shortest_path(graph, *uv, weight=EDGE_TTT_KEY)
            path = node_loc[list(path)].T
            path = path.assign(
                lag=(np.cumsum([ta] + [timedelta(seconds=1) * edge_lag[e] for e in pairwise(path.index)])))

            fig.add_trace(
                go.Scatter3d(
                    x=path.lon,
                    y=path.lat,
                    z=path.lag,
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
            zaxis_title="",
        ),
    )

    fig.write_html(str(out_dir / f"{whatsmyname()}.html"))


def main():
    params = {
        'area': AREA,
        'problem_data_max_trips': 100,
        'where': f"""
            ('2016-05-01 18:00' <= ta) and (ta <= '2016-05-01 18:{TIME_WINDOW:02}')
            and (passenger_count == 1)
        """,

        'num_vehicles': 10,
        'cap_vehicles': 10,
    }

    (graph, trips, routes) = compute_all(**params)
    visualize(graph, trips, routes)


if __name__ == '__main__':
    main()
