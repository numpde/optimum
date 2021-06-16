# RA, 2021-06-13

"""
Try to use `ortools`.
"""

from more_itertools import pairwise
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from itertools import product

from twig import log

import logging

[h.setLevel(logging.DEBUG) for h in log.parent.handlers]

import contextlib

import numpy as np
import pandas as pd
import networkx as nx

from plox import Plox

from opt_utils.graph import odd_king_graph

rng = np.random.default_rng(seed=43)


@contextlib.contextmanager
def draw(graph) -> Plox:
    with Plox() as px:
        pos = nx.get_node_attributes(graph, name='pos')
        nx.draw(graph, ax=px.a, pos=pos)
        nx.draw_networkx_labels(graph, ax=px.a, pos=pos)
        yield px


def make_trips(graph, nmax=64):
    T = 60 * 60 * 24

    # Dream up endpoints of trips
    trips = pd.DataFrame({
        i: rng.choice(graph.nodes, size=nmax, replace=True)
        for i in ['ia', 'ib']
    })

    # Remove trivial trips
    trips = trips[trips.ia != trips.ib]

    # Start time of trip
    trips = trips.assign(ta=rng.uniform(low=0, high=T, size=len(trips)).round())

    log.debug(f"Trips preview (incl. 0): \n{trips.head()}")

    return trips


def reduce(trips, G):
    n2i = pd.Series(sorted({0} | set(trips.ia) | set(trips.ib)))
    i2n = pd.Series(index=n2i.values, data=n2i.index)

    g = nx.DiGraph()
    nx.set_node_attributes(g, n2i.to_dict(), name="i")

    for ((ia, na), (ib, nb)) in product(i2n.items(), repeat=2):
        g.add_edge(na, nb, len=nx.shortest_path_length(G, ia, ib, weight='len'))

    trips = trips.assign(ia=list(i2n[trips.ia])).assign(ib=list(i2n[trips.ib]))

    assert (list(g.nodes) == sorted(list(g.nodes)))

    return (trips, g)


def solve(big_trips, big_graph):
    # (trips, graph) = reduce(big_trips, big_graph)
    (trips, graph) = (big_trips, big_graph)

    assert set(graph.nodes) == set(range(len(graph.nodes)))

    nx.set_node_attributes(graph, {n: n for n in graph.nodes}, name='parent')

    def shadow(i):
        assert i in graph.nodes
        j = len(graph.nodes)

        graph.add_node(j, **{'parent': i})

        for (a, _, d) in graph.in_edges(i, data=True):
            graph.add_edge(a, i, **d)

        for (_, b, d) in graph.out_edges(i, data=True):
            graph.add_edge(i, b, **d)

        eps = 0.1
        graph.add_edge(i, j, len=eps)
        graph.add_edge(j, i, len=eps)

        return j

    trips = trips.assign(ia=trips.ia.apply(shadow))
    trips = trips.assign(ib=trips.ib.apply(shadow))

    # log.debug(nx.get_node_attributes(graph, name='parent'))

    log.debug(f"Trips: \n{trips}")

    data = dict()

    data['dist'] = nx.adjacency_matrix(graph, weight='len').todense()

    # # data['dist'] = nx.floyd_warshall_numpy(graph, weight='len')

    data['dist'] = [
        [
            nx.shortest_path_length(graph, a, b, weight='len')

            # nx.get_edge_attributes(graph, name='len')[(a, b)] if (a, b) in graph.edges
            # else
            # 0 if (a == b)
            # else
            # 1000

            for b in graph.nodes
        ]
        for a in graph.nodes
    ]

    # print(data['dist'])
    # exit(101)

    # data['dist'] = nx.linalg.graphmatrix.adjacency_matrix(graph, weight='len').todense()
    # data['dist'][data['dist'] == 0] = 1e3
    # data['dist'] -= np.triu(np.tril(data['dist']))

    data['dist'] = np.ceil(data['dist']).astype(int)
    # data['dist'] = data['dist'] + data['dist'].T
    # print(data['dist'])

    data['num_vehicles'] = 1
    data['depot'] = 0  # TODO

    data['pickups_deliveries'] = trips[['ia', 'ib']].values

    manager = pywrapcp.RoutingIndexManager(len(data['dist']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def dist(ja, jb):
        return data['dist'][manager.IndexToNode(ja)][manager.IndexToNode(jb)]

    transit_callback_index = routing.RegisterTransitCallback(dist)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # https://developers.google.com/optimization/routing/pickup_delivery

    routing.AddDimension(
        evaluator_index=transit_callback_index,
        slack_max=0,
        capacity=30000,  # vehicle maximum travel distance
        fix_start_cumul_to_zero=True,
        name='distance'
    )

    distance_dimension = routing.GetDimensionOrDie('distance')
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # log.warning("`trips` is ignored.")
    # data['pickups_deliveries'] = [
    #     # [1, 2],
    #     [2, 7],
    #     # [5, 6],
    #     [4, 3],
    #     [5, 6],
    #     # [3, 1],
    # ]

    for (n, (ia, ib)) in enumerate(data['pickups_deliveries']):
        ja = manager.NodeToIndex(ia)
        jb = manager.NodeToIndex(ib)

        log.debug(f"Adding {ia} -> {ib}.")

        routing.AddPickupAndDelivery(ja, jb)
        routing.solver().Add(routing.VehicleVar(ja) == routing.VehicleVar(jb))
        routing.solver().Add(distance_dimension.CumulVar(ja) <= distance_dimension.CumulVar(jb))

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION

    solution = routing.SolveWithParameters(search_parameters)

    log.info(f"Status: {routing.status()}.")

    def print_solution(data, manager, routing, solution):
        # with contextlib.redirect_stdout
        """Prints solution on console."""
        print(f'Objective: {solution.ObjectiveValue()}')
        total_distance = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            route_distance = 0
            while not routing.IsEnd(index):
                n = manager.IndexToNode(index)
                n = nx.get_node_attributes(graph, name='parent')[n]
                plan_output += " {} -> ".format(n)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            plan_output += '{}\n'.format(manager.IndexToNode(index))
            plan_output += 'Distance of the route: {}m\n'.format(route_distance)
            print(plan_output)
            total_distance += route_distance
        print('Total Distance of all routes: {}m'.format(total_distance))

    print_solution(data, manager, routing, solution)
    exit(101)

    def get_routes(solution, routing, manager):
        """Get vehicle routes from a solution and store them in an array."""
        # Get vehicle routes and store them in a two dimensional array whose
        # i,j entry is the jth location visited by vehicle i along its route.
        routes = []
        for route_nbr in range(routing.vehicles()):
            index = routing.Start(route_nbr)
            route = [manager.IndexToNode(index)]
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))
            routes.append(route)
        return routes

    print(get_routes(solution, routing, manager))
    exit()

    # pywrapcp.Ro


def main():
    # graph = odd_king_graph(2, 2)

    graph = nx.DiGraph()
    graph.add_edges_from(pairwise(list(range(6)) + [0]))
    nx.set_edge_attributes(graph, 1, name='len')
    # pos = nx.circular_layout(graph)
    pos = nx.spring_layout(graph, pos=nx.planar_layout(graph, scale=1))
    nx.set_node_attributes(graph, pos, name='pos')

    # randomize edge lengths
    nx.set_edge_attributes(
        graph,
        pd.Series(nx.get_edge_attributes(graph, 'len')) * rng.uniform(0.7, 1.3, size=graph.number_of_edges()),
        name='len'
    )

    # with draw(graph) as px:
    #     px.show()
    #     exit(101)

    # trips = make_trips(graph, nmax=4)
    trips = pd.DataFrame([(1, 4), (2, 3), (4, 2)], columns=['ia', 'ib'])

    trips = trips[(trips != 0).all(axis=1)]  # remove the depot location
    assert len(trips)

    solve(trips, graph)


if __name__ == '__main__':
    main()
