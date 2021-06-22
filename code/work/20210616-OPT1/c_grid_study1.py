# 2021-06-18

import contextlib
import shutil

from twig import log

from pathlib import Path

import pandas as pd

from inclusive import range
from tcga.utils import mkdir, pretty, First, Now, relpath

from b_casestudent import main as worker
from b_casestudent import get_default_params, LOG_FILE

out_dir = mkdir(Path(__file__).with_suffix('') / f"{Now()}")


def run(new_params: dict, work_dir: Path):
    assert work_dir.is_dir()

    params = get_default_params()

    params['data']['focus_radius'] = 1000
    params['data']['max_trips'] = 100
    params['search']['solver_solution_limit'] = 1000

    # TODO: hour for the graph?

    # Setup `params`
    for category in params:
        for variable in params[category]:
            if variable in new_params:
                params[category][variable] = (type(params[category][variable]))(new_params[variable])

    results = worker(out_dir=work_dir, plot=False, **params)


def run_all():

    grid = pd.DataFrame(data=[
        {
            k: v
            for (k, v) in sorted(locals().items()) if not k.startswith('.')
        }
        for repetition in [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for sample_trip_seed in [10 + repetition]

        for bustakers_fraction in [1.0, 0.8, 0.6, 0.4, 0.2]

        for sample_trip_frac in [bustakers_fraction]
        for graph_ttt_factor in [(3 - 2 * bustakers_fraction)]
        for num_vehicles in [10]
        for cap_vehicles in [8]
    ])

    grid.to_csv(out_dir / "param_grid.tsv", sep='\t', index=True, index_label="param_set")

    for (i, row) in grid.iterrows():
        run(row.to_dict(), work_dir=mkdir(out_dir / f"subcases/{i}"))


def main():
    try:
        run_all()
    finally:
        shutil.copyfile(LOG_FILE, out_dir / "casestudent.log")


if __name__ == '__main__':
    main()
