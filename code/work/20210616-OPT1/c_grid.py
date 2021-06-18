# 2021-06-18

import shutil

from twig import log

from pathlib import Path

import pandas as pd

from tcga.utils import mkdir, pretty, First, Now, relpath

from b_casestudy1 import main as worker
from b_casestudy1 import get_default_params, LOG_FILE

out_dir = mkdir(Path(__file__).with_suffix('') / f"{Now()}")


def run(new_params: dict, work_dir: Path):
    assert work_dir.is_dir()

    params = get_default_params()

    # Setup `params`
    for category in params:
        for variable in params[category]:
            if variable in new_params:
                params[category][variable] = (type(params[category][variable]))(new_params[variable])

    try:
        results = worker(out_dir=work_dir, plot=False, **params)
    finally:
        shutil.copyfile(LOG_FILE, out_dir / "worker.log")


def run_all():
    grid = pd.DataFrame(data=[
        {
            k: v
            for (k, v) in sorted(locals().items()) if not k.startswith('.')
        }
        for focus_radius in [1000, 2000]
        for solver_solution_limit in [10000, 1, 10, 100, 1000]
        for max_trips in [10, 100, 200, 400]
        for num_vehicles in [10, 20, 40, 80]
        for cap_vehicles in [1, 8]
    ])

    grid.to_csv(out_dir / "param_grid.tsv", sep='\t', index=True, index_label="param_set")

    for (i, row) in grid.iterrows():
        run(row.to_dict(), work_dir=mkdir(out_dir / f"{i}"))


def main():
    run_all()


if __name__ == '__main__':
    main()
