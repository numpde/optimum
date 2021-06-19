# RA, 2021-06-18

from twig import log

from time import sleep
from pathlib import Path

import pandas as pd

from tcga.utils import relpath, unlist1, mkdir

from opt_utils.misc import Section
from v_visualize import plot_all

index_template = """
<html>
<head><title></title>
<script type='text/javascript' src='https://code.jquery.com/jquery-3.5.1.js'></script>
<script type='text/javascript' src='https://cdn.datatables.net/1.10.25/js/jquery.dataTables.min.js'></script>
<link rel="stylesheet" href="https://cdn.datatables.net/1.10.25/css/jquery.dataTables.min.css">
</head><body>
{table}
<script type='text/javascript'>
$('table').DataTable();
</script>
</body> </html>
"""


def render_index_html(path: Path, grid: pd.DataFrame):
    with Section(f"Generating html for {relpath(path)}", out=log.info):
        more_info = pd.DataFrame(index=grid.index, data=[
            {
                'num_trips': len(pd.read_table(unlist1(path.glob(f"*cases/{i}/trips.*")))),
            }
            for i in grid.index
        ])

        img_links = pd.DataFrame(index=grid.index, data=[
            {
                Path(f).stem: f"<a href='{f}'><img src='{f}' height='30px'></a>"
                for f in [str(f.resolve().relative_to(path)) for f in path.glob(f"plots/{i}/*.png")]
            }
            for i in grid.index
        ])

        grid = pd.concat([grid, more_info, img_links], axis=1)
        html = index_template.format(table=grid.to_html(classes="sortable", border=1, index=False, escape=False))

    return html


def main():
    runs = sorted(Path(__file__).parent.glob("c_grid*/*/param_grid.tsv"))

    for param_grid_file in runs:
        grid = pd.read_table(param_grid_file, index_col=0)
        index_html = (param_grid_file.parent / "index.html")

        if index_html.is_file():
            log.info(f"Found {relpath(index_html)}. Skipping folder.")
            continue

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

        with Section(f"Writing html index to {relpath(index_html)}", out=log.info):
            html = render_index_html(param_grid_file.parent, grid)
            with index_html.open(mode='w') as fd:
                print(html, file=fd)


if __name__ == '__main__':
    main()
