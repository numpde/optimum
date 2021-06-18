# RA, 2021-06-18

from twig import log
from pathlib import Path

import pandas

from v_visualize import plot_all


def main():
    runs = Path(__file__).parent.glob("c_*/*/param_grid.*")

    for run in runs:
        grid = pandas.read_table(run)

        for (i, row) in grid.iterrows():
            log.info(f"RUN {i}: {dict(row)}.")

            path = run.parent / f"{i}"
            plot_all(path)


if __name__ == '__main__':
    main()
