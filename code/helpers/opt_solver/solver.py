# RA, 2021-06-14

import networkx
import pandas


class Solver:
    def __init__(self, setup: dict):
        self.check_setup_minimal(setup)
        pass

    @classmethod
    def check_setup_minimal(cls, setup: dict):
        """
        Check minimal requirements for the problem `setup` dictionary
        that are common to all subclassing solvers.
        """

        assert 'graph' in setup
        assert 'trips' in setup

        graph = setup['graph']
        trips = setup['trips']

        assert type(graph) is networkx.DiGraph
        assert type(trips) is pandas.DataFrame

        assert 'ia' in trips.columns
        assert 'ib' in trips.columns
        assert 'ta' in trips.columns

        assert all(trips['ia'].isin(list(graph.nodes)))
        assert all(trips['ib'].isin(list(graph.nodes)))

        assert networkx.algorithms.components.is_strongly_connected(graph)

    @classmethod
    def check_solution_minimal(cls, setup: dict, solution: dict):
        cls.check_setup_minimal(setup)

        # Important bits from the problem setup

        assert 'graph' in setup
        assert 'trips' in setup
        graph = setup['graph']
        trips = setup['trips']

        assert isinstance(graph, networkx.DiGraph)
        assert isinstance(trips, pandas.DataFrame)

        # Solution bits

        assert 'trajectories' in solution
        assert 'assignment' in solution

        trajectories = solution['trajectories']
        assignment = solution['assignment']

        assert isinstance(trajectories, list)
        assert isinstance(assignment, pandas.DataFrame)

        # Each trip has an assigned trajectory (or NA)
        assert assignment.index.equals(trips.index)

        assert 'j' in assignment.columns  # trajectory ID
        assert 'ta' in assignment.columns  # time of pickup
        assert 'ia' in assignment.columns  # node of pickup
        assert 'tb' in assignment.columns  # time of dropoff
        assert 'ib' in assignment.columns  # node of dropoff

        assert assignment.j.dropna().isin(list(range(len(trajectories))))

        for (n, trajectory) in enumerate(trajectories):
            assert isinstance(trajectory, pandas.DataFrame)
            assert 't' in trajectory.columns  # time
            assert 'i' in trajectory.columns  # node in graph
            assert 'n' in trajectory.columns  # number of people on the bus
            assert trajectory['t'].is_monotonic_increasing
            assert all(trajectory['i'].isin(graph.nodes))




