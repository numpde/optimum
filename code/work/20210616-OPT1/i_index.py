# RA, 2021-06-23


from twig import log

import os.path
import pandas as pd

from pathlib import Path

from tcga.utils import relpath, unlist1, mkdir, Now

from opt_utils.misc import Section, datatable_html


def attach_links(src_dir: Path, grid: pd.DataFrame):
    with Section(f"Attaching links for {relpath(src_dir)}", out=None):
        img_links = pd.DataFrame(index=grid.index, data=[
            {
                Path(f).stem: f"<a href='{f}'><img src='{f}' height='30px'></a>"
                for f in [str(f.resolve().relative_to(src_dir)) for f in src_dir.glob(f"*/{i}/*.png")]
            }
            for i in grid.index
        ])

        etc_links = pd.DataFrame(index=grid.index, data=[
            {
                'subcase': (
                        f"#{i}: " +
                        ", ".join([
                            f"<a href='{f}'>{Path(f).stem}</a>"
                            for f in
                            [str(f.resolve().relative_to(src_dir)) for f in sorted(src_dir.glob(f"*cases/{i}/*.*"))]
                        ])
                )
            }
            for i in grid.index
        ])

        comments = pd.DataFrame(index=grid.index, columns=["comments"], data=[
            " ".join(f.open(mode='r').read() for f in src_dir.glob(f"*cases/{i}/comments.txt")) or "--"
            for i in grid.index
        ])

        grid = pd.concat([grid, img_links, etc_links, comments], axis=1)

    return grid


def main():
    runs = sorted(Path(__file__).parent.glob("c_grid_study*/*/param_grid.tsv"))

    for param_grid_file in runs:
        index_html = (param_grid_file.parent / "index.html")
        log.info(f"Writing {relpath(index_html)}")

        # Look for a post-processed alternative `param_grid`
        defacto_gridfile = ([param_grid_file] + sorted(param_grid_file.parent.glob(f"*/{param_grid_file.name}"))).pop()
        log.info(f"using {relpath(defacto_gridfile)}")
        grid = pd.read_table(defacto_gridfile, index_col=0)

        grid = attach_links(param_grid_file.parent, grid)

        table = \
            f"""
            <p>Generated
            from <a href="{os.path.relpath(defacto_gridfile, index_html.parent)}">{defacto_gridfile.name}</a>
            by <a href="{os.path.relpath(Path(__file__), index_html.parent)}">{Path(__file__).name}</a>
            on {Now()}
            </p>
            """ + \
            grid.to_html(classes="display", table_id="data", border=1, index=False, escape=False)

        html = datatable_html(table, title=param_grid_file.parent.stem)

        with index_html.open(mode='w') as fd:
            print(html, file=fd)


if __name__ == '__main__':
    main()
