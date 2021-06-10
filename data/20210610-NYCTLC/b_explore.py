# RA, 2021-06-11

"""
Based on
https://github.com/numpde/transport/blob/master/pt2pt/20191021-NYCTLC/data_preparation/b_explore_taxidata.py
"""

from twig import log

from pathlib import Path
from sqlite3 import connect

import pandas as pd

from itertools import product, chain

from plox import Plox
from tcga.utils import download, unlist1, from_iterable, mkdir, first, First, whatsmyname

BASE = Path(__file__).parent.resolve()
out_dir = Path(__file__).with_suffix('')

style = {
    'font.size': 8,
    'xtick.major.size': 2,
    'ytick.major.size': 0,
    'xtick.major.pad': 1,
    'ytick.major.pad': 1,
}


def query(sql) -> pd.DataFrame:
    log.debug(f"Query: {sql}.")
    with connect(unlist1(BASE.glob("**/trips.db"))) as con:
        return pd.read_sql_query(sql, con)


def trip_duration_vs_distance(table_name):
    sql = F"""
        SELECT [ta], [tb], [distance] as ds
        FROM [{table_name}]
        WHERE ('2016-05-02 08:00' <= ta) and (tb <= '2016-05-02 08:30')
    """

    df = query(sql)
    df['dt'] = pd.to_datetime(df.tb) - pd.to_datetime(df.ta)

    with Plox(style) as px:
        px.a.scatter(df['ds'], df['dt'].dt.total_seconds(), alpha=0.1, edgecolors='none')
        px.a.set_xlabel("Distance, m")
        px.a.set_ylabel("Time, s")
        px.f.savefig(mkdir(out_dir / f"{whatsmyname()}") / f"{table_name}.png")


if __name__ == '__main__':
    tables = {"green_tripdata_2016-05", "yellow_tripdata_2016-05"}

    for table_name in sorted(tables):
        trip_duration_vs_distance(table_name)
