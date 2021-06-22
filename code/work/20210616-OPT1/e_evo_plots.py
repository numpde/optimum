# RA, 2021-06-22

from twig import log

import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

from plox import Plox, rcParam
from tcga.utils import mkdir, relpath

style = {
    rcParam.Font.size: 14,
    rcParam.Legend.framealpha: 0.5,
}

@contextlib.contextmanager
def plot_evo(grid: pd.DataFrame) -> Plox:
    with Plox(style) as px:
        for (n, ((A, B), grid)) in enumerate(grid.groupby(by=['A', 'B'])):
            c = ["darkgreen", "darkblue", "darkorange", "darkred"][n]

            px.a.plot(np.nan, np.nan, '.',  c=c, ms=10, label=f"(A, B) = {(A, B)}")

            for (i, run) in grid.iterrows():
                px.a.plot(run['bustakers_fraction'], run['b_bar'], '.', c=c, ms=12, alpha=0.8, mec='none')

        (xlim, ylim) = (px.a.get_xlim(), px.a.get_ylim())
        a = min(min(xlim), min(ylim))
        px.a.plot((a, 1), (a, 1), '--', lw=1, c='k', alpha=0.9)  #, label="diagonal")

        # px.a.set_xlim(*xlim)
        # px.a.set_ylim(*ylim)

        px.a.grid(lw=0.3)

        px.a.legend(loc='lower right')

        px.a.set_xlabel("Fraction of bustakers")
        px.a.set_ylabel("Preferred fraction of bustakers")

        yield px


def main():
    grid_files = Path(__file__).parent.glob("**/param_grid.extended.tsv")

    for grid_file in grid_files:
        # if "UTC-20210621-232339" not in str(grid_file):
        #     continue

        grid = pd.read_csv(grid_file, sep='\t')

        try:
            with plot_evo(grid) as px:
                px.f.savefig(mkdir(grid_file.parent / Path(__file__).stem) / "evo.png")

            log.info(f"Success on {relpath(grid_file)}.")
        except:
            log.exception(f"Failed on {relpath(grid_file)}.")


if __name__ == '__main__':
    main()
