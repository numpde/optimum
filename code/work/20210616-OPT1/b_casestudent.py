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

from z_sources import EDGE_TTT_KEY, get_problem_data, postprocess_problem_data

from ortools.constraint_solver.routing_enums_pb2 import FirstSolutionStrategy

# InitGoogleLogging()

from twig import log, LOG_FILE
import logging

log.parent.handlers[0].setLevel(logging.DEBUG)

import contextlib
from typing import List
from pathlib import Path
from datetime import timedelta, datetime
from more_itertools import pairwise

import numpy as np
import pandas as pd
import networkx as nx

import json
import percache

from tqdm import tqdm as progressbar

from geopy.distance import distance as geodistance

from tcga.utils import unlist1, relpath, mkdir, first, Now, whatsmyname, First
from plox import Plox, rcParam

from opt_utils.graph import GraphNearestNode, GraphPathDist
from opt_utils.misc import Section, JSONEncoder

BASE = Path(__file__).parent

cache = percache.Cache(str(mkdir(Path(__file__).with_suffix('')) / f"percache.dat"), livesync=True, repr=repr)
cache.clear(maxage=(60 * 60 * 24 * 7))

sql_string = First(str.split).then(' '.join)

style = {rcParam.Font.size: 14}


class globals:
    out_dir = None


# https://developers.google.com/optimization/routing/routing_options
class OrStatus:
    NOT_YET = 0
    SUCCESS = 1
    FAIL_NOT_FOUND = 2
    FAIL_TIMEOUT = 3
    INVALID = 4


@cache
def reduce_to_clique(problem: pd.Series, hash=None):
    log.debug(f"`{whatsmyname()}` is busy now...")

    n_depot = problem.depot
    assert n_depot in set(problem.graph.nodes)

    # 1. Make clique graph of nodes-of-interest

    # Put `depot` first
    support_nodes = pd.Series([n_depot] + sorted(set(problem.trips.ia) | set(problem.trips.ib)))

    # Note: This may be unnecessary (and a little restrictive)
    assert support_nodes.is_unique

    with Section(f"Computing all-to-all distances", out=log.debug):
        dist = {
            a: pd.Series(nx.single_source_dijkstra_path_length(problem.graph, a, weight=EDGE_TTT_KEY))[support_nodes]
            for a in progressbar(support_nodes)
        }

    graph = nx.from_dict_of_dicts(create_using=nx.DiGraph, d={
        # all-to-all distance matrix
        a: {
            b: {EDGE_TTT_KEY: dist[a][b]}
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

    # Nodes are consecutive integers now (order irrelevant)
    assert set(graph.nodes) == set(range(len(graph.nodes)))

    shadow_eps = 1  # seconds

    def shadow(i):
        assert i in graph.nodes

        # New node
        j = len(graph.nodes)

        graph.add_node(j, **graph.nodes[i], shadows=i)

        # Connect shadow node to parent
        graph.add_edge(i, j, **{'len': 0, EDGE_TTT_KEY: shadow_eps})
        graph.add_edge(j, i, **{'len': 0, EDGE_TTT_KEY: shadow_eps})

        return j

    trips = trips.assign(ia=trips.ia.apply(shadow))
    trips = trips.assign(ib=trips.ib.apply(shadow))

    # 4. All-to-all distance matrix

    shadowed_node = nx.get_node_attributes(graph, name='shadows')
    edge_ttt = nx.get_edge_attributes(graph, name=EDGE_TTT_KEY)

    def dist(i, j):
        (p, q) = (shadowed_node.get(i, i), shadowed_node.get(j, j))
        d = (edge_ttt[(i, p)] if (i != p) and (i != j) else 0) + \
            (edge_ttt[(q, j)] if (q != j) and (i != j) else 0) + \
            (edge_ttt[(p, q)] if (p != q) else 0)
        # log.debug(f"{i}->{j}: {d} / {nx.shortest_path_length(graph, i, j, weight=EDGE_TTT_KEY)}")
        return d

    time_mat = {i: {j: int(dist(i, j)) for j in graph.nodes} for i in graph.nodes}

    # 5. Repackage

    problem = problem.copy()
    problem['graph'] = graph
    problem['trips'] = trips
    problem['original_node'] = nx.get_node_attributes(graph, name=parent_attr)
    problem['time_mat'] = time_mat

    return problem


def solve(problem: pd.Series, **params):
    log.debug(f"`{(whatsmyname())}` is busy now...")

    """
    MEMO:
    
    Rather than changing parameters within this function,
    put them into the input dictionary `params` to increase
    the chances that a responsible cache will be invalidated.
    """

    graph: nx.DiGraph = problem.graph
    trips: pd.DataFrame = problem.trips

    num_vehicles = params['num_vehicles']  # Number of buses
    cap_vehicles = params['cap_vehicles']  # Bus passenger capacity

    depot = 0
    capacities = [cap_vehicles] * num_vehicles

    assert (graph.number_of_nodes() <= 10000), \
        f"The graph seems too large for this code."

    assert (depot in graph.nodes), \
        f"The depot node {depot} not found in the graph."

    assert (depot not in (set(trips.ia) | set(trips.ib))), \
        f"The depot node should not be a trip endpoint."

    assert nx.is_strongly_connected(graph), \
        f"All graph nodes should be reachable from any other."

    assert 'n' in trips.columns, \
        f"Number of passengers 'n' not given."

    # A briefer on variables in `ortools`
    # https://developers.google.com/optimization/reference/constraint_solver/routing

    # Distance should be integers (ortools)
    # https://developers.google.com/optimization/routing/tsp
    if ('time_mat' in problem):
        time_mat = problem['time_mat']
    else:
        with Section(f"`time_mat` dictionary not in `problem` => computing afresh.", out=log.warning):
            time_mat = {
                a: {b: int(nx.shortest_path_length(graph, a, b, weight=EDGE_TTT_KEY)) for b in graph.nodes}
                for a in progressbar(graph.nodes)
            }

    # Beginning of time
    T0 = min(set(trips.ta)) - params['time_buffer']
    rel_t = (lambda t: int((t - T0).total_seconds()))

    # Last timepoint of potential interest
    # T1 = trips[['twa', 'twb']].applymap(np.max).max().max() + TIME_BUFFER
    T1 = T0 + params['time_horizon']  # this seems to make it easier to generate solutions

    # ORTOOLS

    from ortools.constraint_solver.pywrapcp import RoutingIndexManager, RoutingModel
    from ortools.constraint_solver.pywrapcp import DefaultRoutingSearchParameters, DefaultRoutingModelParameters, \
        DefaultPhaseParameters
    from ortools.constraint_solver.pywrapcp import Assignment

    manager = RoutingIndexManager(graph.number_of_nodes(), num_vehicles, depot)
    routing = RoutingModel(manager)

    # Travel time

    def time_callback(ja, jb):
        return time_mat[manager.IndexToNode(ja)][manager.IndexToNode(jb)]
        # return nx.shortest_path_length(graph, manager.IndexToNode(ja), manager.IndexToNode(jb), weight=EDGE_TTT_KEY)

    ix_transit_time = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(ix_transit_time)

    routing.AddDimension(
        evaluator_index=ix_transit_time,
        name='travel_time',
        # allow waiting time:
        slack_max=int(params['max_vehicle_waiting_time'].total_seconds()),
        # vehicle maximum travel time:
        capacity=int((T1 - T0).total_seconds()),
        # the start time is a window specified below
        fix_start_cumul_to_zero=False,
    )

    dim_travel_time = routing.GetDimensionOrDie('travel_time')
    dim_travel_time.SetGlobalSpanCostCoefficient(params['span_cost_coeff_travel_time'])

    # Register all slack variables with the `assignment` -- ?

    # for i in graph.nodes:
    #     routing.AddToAssignment(dim_travel_time.SlackVar(manager.NodeToIndex(i)))

    # Vehicle start time window

    for iv in range(num_vehicles):
        dim_travel_time.CumulVar(routing.Start(iv)).SetRange(rel_t(T0), rel_t(T1))
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
        # log.debug(f"{t0}, {trip.twa}, {list(map(rel_t, trip.twa))}")
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

    # log.debug(f"Demands: {demands[demands != 0]}")

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

    vehicle_capacity = routing.GetDimensionOrDie('vehicle_capacity')
    vehicle_capacity.SetGlobalSpanCostCoefficient(params['span_cost_coeff_vehicle_capacity'])

    # Allow to drop nodes

    penalty_unserviced = params['penalty_unserviced']

    for node in demands[demands != 0].index:
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty_unserviced)

    # Initialize solver parameters

    # https://developers.google.com/optimization/reference/constraint_solver/routing
    search_params = DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = params['first_solution_strategy']

    # https://stackoverflow.com/questions/63600059/how-can-i-get-progress-log-on-google-or-tools
    search_params.log_search = True

    # search_params.routing_no_lns = True  # no such parameter
    # search_params.routing_guided_local_search = True  # no such parameter
    #
    search_params.time_limit.seconds = int(params['solver_time_limit'].total_seconds())
    search_params.solution_limit = params['solver_solution_limit']

    # Solve

    with Section("Running `ortools` solver", out=log.info):
        assignment: Assignment = routing.SolveWithParameters(search_params)

    log.info(f"Assignment: {str(assignment)[0:40]}...")
    log.info(f"Routing status: {routing.status()}")
    log.info(f"Solver success: {routing.status() == OrStatus.SUCCESS}")

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
                    yield (i, j, assignment.Value(vehicle_capacity.CumulVar(j)), slack)
                    if routing.IsEnd(j):
                        break
                    else:
                        j = assignment.Value(routing.NextVar(j))

            route = pd.DataFrame(data=get_route_from(routing.Start(iv)), columns=['i', 'j', 'load', 'slack'])

            route = route.assign(cost=np.cumsum(
                [0] + [routing.GetArcCostForVehicle(*e, iv) for e in pairwise(route.i)]
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
            T0 + timedelta(seconds=1) * assignment.Value(dim_travel_time.CumulVar(j))
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
                ] + [
                    np.nan
                ]
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
                    # log.debug(f"\n{route.head(3)}")
                    trips.loc[i, 'iv'] = iv
                    trips.loc[i, 'iv_ta'] = first(route.est_time_dep)
                    trips.loc[i, 'iv_tb'] = route.est_time_arr[unlist1(route[route.i == trip.ib].index)]

    # log.info(f"Solution: \n{trips.sort_values(by=['iv', 'iv_ta']).to_markdown()}")

    # As one dataframe
    routes = pd.concat([route.assign(iv=iv) for (iv, route) in enumerate(routes)])

    return (trips, routes)



@cache
def compute_all(**params):
    problem_data = get_problem_data(**params['data'])

    problem_data = postprocess_problem_data(problem_data, **params['data_post'])

    reduced_problem_data = reduce_to_clique(problem_data, hash={**params['data'], **params['data_post']})
    (trips, routes) = solve(reduced_problem_data, **params['fleet'], **params['optimization'], **params['search'])

    trips = trips.assign(ia=trips.ia.map(reduced_problem_data.original_node))
    trips = trips.assign(ib=trips.ib.map(reduced_problem_data.original_node))

    routes = routes.assign(i=routes.i.map(reduced_problem_data.original_node))

    # log.info(f"Solution: \n{trips.sort_values(by=['iv', 'iv_ta']).to_markdown()}")

    return (trips, routes)


def main(out_dir=None, **params):
    try:
        log.debug(f"{Path(__file__).name} output folder: {relpath(out_dir)}")

        with (out_dir / "params.json").open(mode='w') as fd:
            print(json.dumps(params, indent=2, cls=JSONEncoder), file=fd)

        globals.out_dir = out_dir
        (trips, routes) = compute_all(**params)

        trips.to_csv(out_dir / "trips.tsv", sep='\t')
        routes.to_csv(out_dir / "routes.tsv", sep='\t')
    except:
        log.exception(f"`main` solution failed.")
        raise

    return locals()


def get_default_params():
    # DON'T CHANGE ANY HERE
    return {
        'data': {
            'table_names': sorted({"green_tripdata_2016-05", "yellow_tripdata_2016-05"}),
            'area': "manhattan",
            'max_trips': 100,
            'sql_where': sql_string(f"""
                ('2016-05-01 18:00' <= ta) and 
                (tb <= '2016-05-01 19:00') and
                (passenger_count == 1)
            """),
            'focal_point': (40.75798, -73.98550),  # Times Square
            'focus_radius': 1000,

            # lag-graph hour
            'graph_h': 18,
        },

        'data_post': {
            'sample_trip_frac': 1.0,
            'sample_trip_seed': 43,

            'graph_ttt_factor': 1.0,
        },

        'fleet': {
            'num_vehicles': 10,
            'cap_vehicles': 1,
            'max_vehicle_waiting_time': timedelta(minutes=10),
        },

        'optimization': {
            'penalty_unserviced': 10_000,  # seconds (presumably)
            'span_cost_coeff_travel_time': 0,
            'span_cost_coeff_vehicle_capacity': 0,
        },

        'search': {
            # https://developers.google.com/optimization/routing/routing_options
            'first_solution_strategy': FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,

            'solver_time_limit': timedelta(minutes=60),
            'solver_solution_limit': 1000,
            'time_buffer': timedelta(minutes=10),
            'time_horizon': timedelta(days=10),
        },
    }


if __name__ == '__main__':
    out_dir = mkdir(Path(__file__).with_suffix('') / f"{Now()}").resolve()
    main(out_dir=out_dir, **get_default_params())

    from v_visualize import plot_all

    plot_all(out_dir, out_dir)
