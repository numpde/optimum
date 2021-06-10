# RA, 2021-06-10

"""
Based on
https://github.com/numpde/transport/blob/master/pt2pt/20191021-NYCTLC/data_preparation/a_download_taxidata.py
"""

from twig import log

import json
import sqlite3

import pandas as pd

from itertools import product, chain
from pathlib import Path

import shapely.geometry
import shapely.prepared

from tcga.utils import download, unlist1, from_iterable, mkdir, first, First

METERS_PER_MILE = 1609.34

URLS = {
    "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-05.csv",
    "https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2016-05.csv",
}

PARAM = pd.Series({
    'max_trip_distance_miles': 30,
    'min_trip_distance_miles': 0.1,
})

BASE = Path(__file__).parent.resolve()

download = download.to(abs_path=(BASE / "download_cache"))


def download_all():
    for url in URLS:
        log.info(f"Downloading {Path(url).name}.")
        log.info(f"OK: {download(url).now.meta}.")


def get_geoselector():
    # Selector based on a region (e.g. Manhattan)
    # https://shapely.readthedocs.io/en/stable/manual.html#object.contains
    with unlist1(BASE.parent.glob("**/manhattan.geojson")).open(mode='r') as fd:
        bounding_shape = max(from_iterable(shapely.geometry.shape(json.load(fd))), key=(lambda p: p.length))
        return shapely.prepared.prep(bounding_shape).contains


def subset_to_selector(df, selector=get_geoselector()) -> pd.DataFrame:
    df = df.loc[list(map(selector, map(shapely.geometry.Point, zip(df.xa_lon, df.xa_lat)))), :]
    df = df.loc[list(map(selector, map(shapely.geometry.Point, zip(df.xb_lon, df.xb_lat)))), :]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase column names
    df.columns = map(str.strip, map(str.lower, df.columns))

    # Remove prefixes from pickup/dropoff datetimes
    for e in ["pickup_datetime", "dropoff_datetime"]:
        df.columns = map(lambda c: (e if c.endswith(e) else c), df.columns)

    df.columns = map(lambda c: c.replace("latitude", "lat"), df.columns)
    df.columns = map(lambda c: c.replace("longitude", "lon"), df.columns)

    df = df.rename(columns={"pickup_datetime": "ta"})
    df = df.rename(columns={"dropoff_datetime": "tb"})

    df['duration'] = (pd.to_datetime(df['tb']) - pd.to_datetime(df['ta'])).dt.total_seconds()

    df = df.rename(columns={"pickup_lat": "xa_lat"})
    df = df.rename(columns={"pickup_lon": "xa_lon"})
    df = df.rename(columns={"dropoff_lat": "xb_lat"})
    df = df.rename(columns={"dropoff_lon": "xb_lon"})

    # Omit rows with bogus lat/lon entries
    for c in map("_".join, product(["xa", "xb"], ["lat", "lon"])):
        df = df[df[c] != 0]

    # Omit rows with small/large trip distance (in miles)
    df = df[df.trip_distance >= PARAM.min_trip_distance_miles]
    df = df[df.trip_distance <= PARAM.max_trip_distance_miles]

    # Convert travel distance to meters
    df['trip_distance'] *= METERS_PER_MILE
    df = df.rename(columns={"trip_distance": "distance"})

    return df


def write_to_db():
    for url in sorted(URLS):
        table_name = Path(url).stem
        log.info(f"Writing table `{table_name}`.")

        with sqlite3.connect(mkdir(BASE / "sqlite") / "trips.db") as con:
            con.cursor().execute(F"DROP TABLE IF EXISTS [{table_name}]")

            with download(url).now.open(mode='r') as fd:
                for df in pd.read_csv(fd, chunksize=(1024 * 1024)):
                    df = clean(df)
                    df = subset_to_selector(df)

                    df.to_sql(name=table_name, con=con, if_exists='append', index=False)


if __name__ == '__main__':
    download_all()
    write_to_db()
