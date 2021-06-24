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
    rcParam.Font.size: 20,
    rcParam.Legend.framealpha: 0.5,
}


@contextlib.contextmanager
def plot_evo(grid: pd.DataFrame) -> Plox:
    if not {'A', 'B'}.issubset(grid):
        log.warning(f"Columns `A` and `B` not found -- skipping.")
        yield None
        return

    with Plox(style) as px:
        for (n, ((A, B), grid)) in enumerate(grid.groupby(by=['A', 'B'])):
            c = ["green", "darkblue", "darkorange", "darkred"][n]
            s = ["o", "s", "o", "s"][n]

            px.a.plot(np.nan, np.nan, s, c=c, ms=12, label=f"A = {A}\,\$, B = {B}\,\$")

            for (i, run) in grid.iterrows():
                b0 = run['bustakers_fraction'] * 100
                b1 = b0 * (1 - run['unserviced'] / run['num_trips'])
                b_bar = run['b_bar'] * 100
                px.a.plot(b0, b_bar, s, c=c, ms=10, alpha=0.2, mec='none')
                px.a.plot(b1, b_bar, s, c=c, ms=10, alpha=0.7, mec='none')
                px.a.plot([b0, b1], [b_bar, b_bar], '-', c=c, lw=0.5, alpha=0.1)

        (xlim, ylim) = (px.a.get_xlim(), px.a.get_ylim())
        # a = (10 * int(min(min(xlim), min(ylim)) / 10))
        a = 10
        px.a.plot((a, 100), (a, 100), '--', lw=1, c='k', alpha=0.9)

        px.a.set_xticks(np.arange(a, 101)[0::10])
        px.a.set_xticklabels([f"{x}\%" for x in px.a.get_xticks()], fontsize="small")

        px.a.set_yticks(px.a.get_xticks())
        px.a.set_yticklabels([f"{x}\%" for x in px.a.get_yticks()], fontsize="small")

        px.a.grid(lw=0.3)

        px.a.legend(loc='lower right')

        px.a.set_xlabel("Take the minibus")
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

            if "c_grid_study1" in [p.name for p in grid_file.parents]:
                split = []
            else:
                split = sorted({'graph_h', 'graph_ttt_factor'}.intersection(grid.columns))

            for (key, df) in (grid.groupby(by=split) if split else [([], grid)]):
                stem = "__".join(["evo"] + [f"{k}={v}" for (k, v) in zip(split, key if isinstance(key, tuple) else [key])])
                with plot_evo(df) as px:
                    px.f.savefig(mkdir(grid_file.parent.parent / Path(__file__).stem) / f"{stem}.png")
        except:
            log.exception(f"Failed on {relpath(grid_file)}.")
        else:
            log.info(f"Success on {relpath(grid_file)}.")


if __name__ == '__main__':
    main()
