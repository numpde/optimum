# RA, 2021-06-18

from twig import log

from time import sleep
from pathlib import Path

import pandas as pd

from tcga.utils import relpath, unlist1, mkdir

from opt_utils.misc import Section
from v_visualize import plot_all

# https://datatables.net/extensions/fixedheader/examples/options/columnFiltering.html
index_template = r"""
<html>
<head><title></title>
<script type="text/javascript" src="https://code.jquery.com/jquery-3.5.1.js"></script>
<script type="text/javascript" src="https://cdn.datatables.net/1.10.25/js/jquery.dataTables.min.js"></script>
<script type="text/javascript" src="https://cdn.datatables.net/fixedheader/3.1.9/js/dataTables.fixedHeader.min.js"></script>
<link rel="stylesheet" href="https://cdn.datatables.net/1.10.25/css/jquery.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/fixedheader/3.1.9/css/fixedHeader.dataTables.min.css">
<style>
thead input {{ width: 50px; }}
</style>
</head><body>
<p>
For an exact-match column search use regex, e.g.: ^100$
</p>

{table}

<script type="text/javascript">
$(document).ready(function() {
    // Setup - add a text input to each footer cell
    $('#data thead tr').clone(true).appendTo( '#data thead' );
    $('#data thead tr:eq(1) th').each( function (i) {
        var title = $(this).text();
        $(this).html('<input type="text" placeholder="Search ' + title + '" />');
 
        $('input', this).on('keyup change', function () {
            if (table.column(i).search() !== this.value) {
                // https://stackoverflow.com/questions/44003585/datatables-column-search-for-exact-match
                table.column(i).search(this.value, true, false).draw();
            }
        });
        
        $('#data > thead > tr > th').css({'width': '100px'});
        $('#data input').css({'width': '100%'});
        $('#data').css({'width': '100%'});
    } );
 
    var table = $('#data').DataTable({
        orderCellsTop: true,
        fixedHeader: true,
        "lengthMenu": [[-1, 8, 16, 32, 64], ["All", 8, 16, 32, 64]],
        "pageLength": 16,
    });
});
</script>
</body></html>
"""


def render_index_html(path: Path, grid: pd.DataFrame):
    with Section(f"Generating html for {relpath(path)}", out=log.info):
        def read_table(subpath: str):
            try:
                return pd.read_table(unlist1(path.glob(subpath)))
            except ValueError:
                return pd.DataFrame()

        more_info = pd.DataFrame(index=grid.index, data=[
            {
                'num_trips': len(read_table(f"*cases/{i}/trips.*")),
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

        etc_links = pd.DataFrame(index=grid.index, data=[
            {
                'subcase': (
                    f"#{i}: " +
                    ", ".join([
                        f"<a href='{f}'>{Path(f).stem}</a>"
                        for f in [str(f.resolve().relative_to(path)) for f in sorted(path.glob(f"*cases/{i}/*.*"))]
                    ])
                )
            }
            for i in grid.index
        ])

        grid = pd.concat([grid, more_info, img_links, etc_links], axis=1)
        # grid = grid.reset_index()
        table = grid.to_html(classes="display", table_id="data", border=1, index=False, escape=False)
        html = index_template.replace("{table}", table)

    return html


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

        with Section(f"Writing html index to {relpath(index_html)}", out=log.info):
            html = render_index_html(param_grid_file.parent, grid)
            with index_html.open(mode='w') as fd:
                print(html, file=fd)


if __name__ == '__main__':
    main()
