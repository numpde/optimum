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
    (a_early, a_late) = (timedelta(minutes=3), timedelta(minutes=10))
    (b_early, b_late) = (timedelta(minutes=20), timedelta(minutes=15))
    trips['twa'] = trips.ta.apply(lambda t: (t - a_early, t + a_late))
    trips['twb'] = trips.tb.apply(lambda t: (t - b_early, t + b_late))
    return trips


def concentrated_subset(trips: pd.DataFrame):
    max_dist = 2000  # meters

    dist_to_times_square = pd.Series(index=trips.index, data=[
        max(geodistance(row.xa, TIMES_SQUARE).m, geodistance(row.xb, TIMES_SQUARE).m)
        for (i, row) in trips.iterrows()
    ])

    trips = trips[dist_to_times_square <= max_dist]
    return trips


def formulate_problem():
    area = AREA

    where = f"""
        ('2016-05-01 18:00' <= ta) and (ta <= '2016-05-01 18:{TIME_WINDOW:02}')
        and (passenger_count == 1)
    """

    trips = pd.concat(axis=0, objs=[
        get_raw_trips(table_name, where=where).assign(table_name=table_name)
        for table_name in sorted(table_names)
    ])

    assert len(trips), \
        f"Query returned zero trips. Maybe a misshappen `where`: \n{where}"

    trips = trips[list(KEEP_COLS)].sort_values(by='ta')
    trips = trips.assign(xa=list(zip(trips.xa_lat, trips.xa_lon)))
    trips = trips.assign(xb=list(zip(trips.xb_lat, trips.xb_lon)))

    log.debug(f"Trips: \n{trips.head(3).to_markdown()} \netc.")

    graph = largest_component(load_graph(area))
    trips = with_nearest_ingraph(trips, graph)

    trips = attach_timewindows(trips)

    # Simplify problem
    trips = concentrated_subset(trips)

    # TODO: remove
    trips = trips.head(15)
    log.warning(f"Reducing the number of trips to {len(trips)}.")

    with plot_trajectories(graph, trips) as px:
        px.f.savefig(out_dir / f"bare_trajectories.png")

    return pd.Series({'area': area, 'table_names': table_names, 'graph': graph, 'trips': trips})


def reduce_to_clique(problem):
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

    i2n = nx.get_node_attributes(graph, name=parent_attr)
    n2i = {o: i for (i, o) in i2n.items()}

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

    return pd.Series({'graph': graph, 'trips': trips})


def solve(problem):
    graph: nx.DiGraph = problem.graph
    trips: pd.DataFrame = problem.trips

    depot = 0
    num_vehicles = 5
    capacities = [10] * num_vehicles

    assert (graph.number_of_nodes() <= 1000), \
        "The graph seems too large for this code."

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
    from ortools.constraint_solver.routing_enums_pb2 import FirstSolutionStrategy
    from ortools.constraint_solver.pywrapcp import Assignment

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

    # Vehicle start time window

    for iv in range(num_vehicles):
        dim_travel_time.CumulVar(routing.Start(iv)).SetRange(rel_t(t0), rel_t(t0 + timedelta(hours=3)))

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

    # Demands & capacities
    # https://developers.google.com/optimization/routing/penalties

    demands = trips[['ia', 'n']].set_index('ia').n.reindex(graph.nodes).fillna(0).astype(int)
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

    # Allow to drop nodes

    penalty_unserviced = 1000  # TODO:?

    for node in demands[demands != 0].index:
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty_unserviced)

    # Initialize solver parameters

    params = DefaultRoutingSearchParameters()
    params.first_solution_strategy = FirstSolutionStrategy.PATH_CHEAPEST_ARC
    # params.time_limit.seconds = 30  # TODO
    params.solution_limit = 100  # TODO

    # Solve

    with Section("Running `ortools` solver", out=log.info):
        assignment: Assignment = routing.SolveWithParameters(params)

    print(assignment)
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

    def get_routes(manager, routing, assignment):
        for iv in range(manager.GetNumberOfVehicles()):
            def get_route_from(j):
                while True:
                    yield manager.IndexToNode(j)
                    if routing.IsEnd(j):
                        break
                    else:
                        j = assignment.Value(routing.NextVar(j))

            route = pd.DataFrame(data={'i': list(get_route_from(routing.Start(iv)))})

            route = route.assign(cost=np.cumsum(
                [0] +
                [routing.GetArcCostForVehicle(*e, iv) for e in pairwise(route.i)]
            ))

            yield route

    routes = list(get_routes(manager, routing, assignment))

    def cum_path_time(graph, segments):
        # Typically: cum_path_length(graph, pairwise(path))
        ts = np.cumsum([0] + [nx.shortest_path_length(graph, *seg, weight=EDGE_TTT_KEY) for seg in segments])
        return t0 + ts * timedelta(seconds=1)

    routes = [
        route.assign(time=(
            timedelta(seconds=assignment.Value(dim_travel_time.CumulVar(routing.Start(iv)))) +
            cum_path_time(graph, pairwise(route.i))
        ))
        for (iv, route) in enumerate(routes)
    ]

    for route in routes:
        # log.info(f"Vehicle route: \n{route}")
        log.info(f"Vehicle route: \n{route.assign(add_load=list(demands[route.i]))}")

    # Infer pickups and deliveries

    trips = trips.assign(iv=np.nan, iv_ta=np.nan, iv_tb=np.nan)

    for (i, trip) in trips.iterrows():
        if trip.ia not in unserviced:
            for (iv, route) in enumerate(routes):
                if trip.ia in set(route.i):
                    route = route.loc[unlist1(route[route.i == trip.ia].index):]
                    log.debug(f"\n{route.head(3)}")
                    trips.loc[i, 'iv'] = iv
                    trips.loc[i, 'iv_ta'] = first(route.time)
                    trips.loc[i, 'iv_tb'] = route.time[unlist1(route[route.i == trip.ib].index)]

    log.info(f"Solution: \n{trips.sort_values(by=['iv', 'iv_ta']).to_markdown()}")

    # Visualize




def main():
    problem = formulate_problem()
    problem = reduce_to_clique(problem)
    log.info(f"Number of trips: {len(problem.trips)}.")

    solve(problem)


if __name__ == '__main__':
    main()
