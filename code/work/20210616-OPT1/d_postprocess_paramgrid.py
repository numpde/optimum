# RA, 2021-06-21


from twig import log

import os.path

from time import sleep
from pathlib import Path

from sigfig import round

import scipy.stats
import numpy as np
import pandas as pd

from tcga.utils import relpath, unlist1, mkdir, Now

from opt_utils.misc import Section, datatable_html
from opt_utils.misc import Memo

memo = Memo()

(sigma, mu) = memo(0.7, np.round(np.log(6e4)))
income: scipy.stats.rv_continuous = scipy.stats.lognorm(s=sigma, scale=np.exp(mu))


def attach_trivia(src_dir: Path, grid: pd.DataFrame):
    def read_table(subpath: str):
        try:
            return pd.read_table(unlist1(src_dir.glob(subpath)))
        except ValueError:
            return pd.DataFrame()

    trivia = pd.DataFrame(index=grid.index, data=[
        {
            'num_trips': len(read_table(f"*cases/{i}/trips.*")),
        }
        for i in grid.index
    ])

    grid = pd.concat([grid, trivia], axis=1)

    return grid


def attach_predictions(src_dir: Path, grid: pd.DataFrame):
    def rowwise(grid, A, B):
        for (i, row) in grid.iterrows():
            try:
                trips_file = unlist1(src_dir.glob(f"*cases/{i}/trips.*"))
            except ValueError:
                log.warning(f"Case {i} not found. Continuing.")
                yield {}
                continue

            trips = pd.read_table(trips_file, index_col=0, parse_dates=['iv_ta', 'iv_tb'])
            trips = trips.assign(trip_time=(trips.iv_tb - trips.iv_ta).dt.total_seconds())
            trips = trips.assign(excess_time=(trips.trip_time - trips.quickest_time))
            # log.info(f"Trips for case {i}: \n{trips[['quickest_time', 'trip_time', 'excess_time']].to_markdown()}")

            unserviced = sum(pd.isna(trips.excess_time))
            trips = trips[~pd.isna(trips.excess_time)]

            e = np.mean(trips.excess_time)
            c = (A - B) / e * 60 * 60  # $/h
            c *= (52 * 5 * 8)  # $/year

            yield {
                'A': A,
                'B': B,
                'unserviced': unserviced,
                'mean_excess_time': round(float(e), decimals=1),
                'critical_income': round(float(c), sigfigs=4),
                'a_bar': round(float(income.sf(c)), decimals=3),
                'b_bar': round(float(income.cdf(c)), decimals=3),
            }

    with Section(f"Attaching predictions for {relpath(src_dir)}", out=log.info):
        grid = pd.concat(axis=0, objs=[
            pd.concat(axis=1, objs=[grid, pd.DataFrame(index=grid.index, data=rowwise(grid, A=A, B=B))])
            for (A, B) in [(6, 3), (6, 0)]
        ])

    return grid


def main():
    runs = sorted(Path(__file__).parent.glob("c_grid_study*/*/param_grid.tsv"))

    options = {str(i): run for (i, run) in enumerate(runs)}

    if not options:
        print("Nothing to do.")
        return
    elif (len(options) == 1):
        pass
    else:
        print(*[f"{i}: {relpath(options[i])}" for i in options], sep='\n')

        i = input("Which one (`a` for all)? \n").lower().strip()

        if (i == "a"):
            pass
        elif (i in options):
            runs = [options[i]]
        else:
            print("aborting")
            return

        sleep(1)

    for param_grid_file in runs:
        grid = pd.read_table(param_grid_file, index_col=0)

        try:
            grid = attach_trivia(param_grid_file.parent, grid)
        except:
            log.exception(f"`attach_trivia` failed.")

        try:
            grid = attach_predictions(param_grid_file.parent, grid)
        except:
            log.exception(f"`attach_predictions` failed.")

        grid.to_csv(mkdir(param_grid_file.parent / Path(__file__).stem) / param_grid_file.name, sep='\t')


if __name__ == '__main__':
    main()
