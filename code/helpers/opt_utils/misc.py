# RA, 2021-06-11
# RA, 2021-06-20

import json
import datetime
import pathlib
import time
import inspect
import sorcery
import executing

REPO = next(p for p in pathlib.Path(__file__).parents for p in p.glob("code")).parent.resolve()


class Section:
    def __init__(self, desc, out=print):
        co = inspect.currentframe().f_back.f_code.co_name
        (self.desc, self.out) = (f"{co}: {desc}", out)

    def __enter__(self):
        self.out and self.out(f"<{self.desc}>")
        time.sleep(1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.out and self.out(f"[Done]")
        return


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.timedelta):
            return repr(obj)
        elif isinstance(obj, pathlib.Path):
            return str(obj.resolve().relative_to(REPO))
        else:
            return json.JSONEncoder.default(self, obj)


class Memo(dict):
    """
    Example:
        memo = Memo()
        (name, surname) = memo("X", "Y")
        print(memo)  # {'name': 'X', 'surname': 'Y'}

    Modified from `sorcery.spell` and `sorcery.assigned_names`.
    """

    def __call__(self, value0, *values):
        exe = executing.Source.executing(inspect.currentframe().f_back)
        names = sorcery.core.FrameInfo(exe).assigned_names(allow_one=True)[0]

        assert len(names) == (1 + len(values))

        for (name, value) in zip(names, (value0, *values)):
            self[name] = value

        if values:
            return (value0, *values)
        else:
            return value0


def datatable_html(table_html: str):
    # https://datatables.net/extensions/fixedheader/examples/options/columnFiltering.html
    template = r"""
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
    For an exact-match column search use regex, e.g. <b>^100$</b> or <b>10|20</b>
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

    return template.replace("{table}", table_html)
