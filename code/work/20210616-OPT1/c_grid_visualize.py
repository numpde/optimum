# RA, 2021-06-18

from twig import log

from time import sleep
from pathlib import Path

import pandas as pd

from tcga.utils import relpath, unlist1, mkdir

from opt_utils.misc import Section
from v_visualize import plot_all
from opt_utils.misc import datatable_html



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

        if index_html.is_file():
            log.info(f"Found {relpath(index_html)}. Continuing anyway.")
            sleep(3)

        with Section(f"Writing plots for {relpath(param_grid_file)}", out=log.info):
            try:
                for (i, row) in grid.iterrows():
                    log.info(f"{i}: {dict(row)}")
                    plot_all(
                        path_src=unlist1(param_grid_file.parent.glob(f"*cases/{i}")),
                        path_dst=mkdir(param_grid_file.parent / f"plots/{i}"),
                    )
            except:
                log.exception(f"Some plots failed. Continuing...")
                sleep(3)

        # with Section(f"Writing html index to {relpath(index_html)}", out=log.info):
        #     grid = attach_stuff(param_grid_file.parent, grid)
        #
        #     table = grid.to_html(classes="display", table_id="data", border=1, index=False, escape=False)
        #     html = datatable_html(table)
        #
        #     with index_html.open(mode='w') as fd:
        #         print(html, file=fd)


if __name__ == '__main__':
    main()
