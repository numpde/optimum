# RA, 2021-06-22

from twig import log

import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

from plox import Plox, rcParam
from tcga.utils import mkdir, relpath

style = {
    rcParam.Text.usetex: True,
    rcParam.Font.size: 16,
    rcParam.Legend.framealpha: 0.5,
}

@contextlib.contextmanager
def plot_evo(grid: pd.DataFrame) -> Plox:
    with Plox(style) as px:
        for (n, ((A, B), grid)) in enumerate(grid.groupby(by=['A', 'B'])):
            c = ["green", "darkblue", "darkorange", "darkred"][n]

            px.a.plot(np.nan, np.nan, '.',  c=c, ms=14, label=f"A = {A}\,\$, B = {B}\,\$")

            for (i, run) in grid.iterrows():
                b_frac = run['bustakers_fraction'] * (1 - run['unserviced'] / run['num_trips'])
                b_bar = run['b_bar']

                px.a.plot(b_frac * 100, b_bar * 100, '.', c=c, ms=16, alpha=0.8, mec='none')

        (xlim, ylim) = (px.a.get_xlim(), px.a.get_ylim())
        a = int(10 ** np.floor(np.log10(min(min(xlim), min(ylim)))))
        px.a.plot((a, 100), (a, 100), '--', lw=1, c='k', alpha=0.9)

        px.a.set_xticks(np.arange(a, 101)[0::10])
        px.a.set_xticklabels([f"{x}\%" for x in px.a.get_xticks()], fontsize="small")

        px.a.set_yticks(px.a.get_xticks())
        px.a.set_yticklabels([f"{x}\%" for x in px.a.get_yticks()], fontsize="small")

        px.a.grid(lw=0.3)

        px.a.legend(loc='lower right')

        px.a.set_xlabel("Take the minibus (imposed)")
        px.a.set_ylabel("Prefer the minibus")

        yield px


def main():
    grid_files = Path(__file__).parent.glob("**/d_post*/param_grid.tsv")

    for grid_file in grid_files:
        grid = pd.read_csv(grid_file, sep='\t')

        try:
            # bogus datapoint
            if "UTC-20210621-232339" in [p.name for p in grid_file.parents]:
                grid = grid[~grid.param_set.isin([16])]

            with plot_evo(grid) as px:
                px.f.savefig(mkdir(grid_file.parent.parent / Path(__file__).stem) / "evo.png")

            log.info(f"Success on {relpath(grid_file)}.")
        except:
            log.exception(f"Failed on {relpath(grid_file)}.")


if __name__ == '__main__':
    main()
