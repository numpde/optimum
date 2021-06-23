# RA, 2021-06-18

from twig import log

from time import sleep
from pathlib import Path

import pandas as pd

from tcga.utils import relpath, unlist1, mkdir

from opt_utils.misc import Section
from v_visualizer import plot_all


def main():
    runs = sorted(Path(__file__).parent.glob("c_grid*/*/param_grid.tsv"))

    options = {str(i): run for (i, run) in enumerate(runs)}
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
        index_html = (param_grid_file.parent / "index.html")

        with Section(f"Writing plots for {relpath(param_grid_file)}", out=log.info):
            try:
                for (i, row) in grid.iterrows():
                    log.info(f"{i}: {dict(row)}")
                    plot_all(
                        path_src=unlist1(param_grid_file.parent.glob(f"*cases/{i}")),
                        path_dst=mkdir(param_grid_file.parent / f"{Path(__file__).stem}/{i}"),
                    )
            except:
                log.exception(f"Some plots failed. Continuing...")
                sleep(3)
                continue


if __name__ == '__main__':
    main()
