# RA, 2021-06-11

"""
Based on
https://github.com/numpde/transport/blob/master/pt2pt/20191021-NYCTLC/data_preparation/b_explore_taxidata.py
"""

import calendar
from collections import Counter

from twig import log

from pathlib import Path
from sqlite3 import connect

import numpy as np
import pandas as pd

from itertools import product, chain, groupby

from plox import Plox, rcParam
from tcga.utils import download, unlist1, from_iterable, mkdir, first, First, whatsmyname

import z_maps as maps

BASE = Path(__file__).parent.resolve()
out_dir = Path(__file__).with_suffix('')

style = {
    rcParam.Text.usetex: True,

    # rcParam.Font.size: 3,

    rcParam.Font.size: 14,
    rcParam.Axes.labelsize: "large",

    rcParam.Xtick.Major.size: 2,
    rcParam.Ytick.Major.size: 0,

    rcParam.Xtick.Major.pad: 1,
    rcParam.Ytick.Major.pad: 1,
}


def name2color(table_name):
    if "green" in table_name: return "green"
    if "yello" in table_name: return "darkorange"
    return None


def name2cmap(table_name):
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list('company', ["white", name2color(table_name)])


def QUERY(sql) -> pd.DataFrame:
    log.debug(f"Query: {sql}.")
    with connect(unlist1(BASE.glob("**/trips.db"))) as con:
        return pd.read_sql_query(sql, con)


def trip_distance_histogram(table_name):
    sql = F"SELECT [distance] FROM [{table_name}]"
    trip_distance = QUERY(sql).squeeze()

    assert isinstance(trip_distance, pd.Series)

    # Convert to km
    trip_distance *= (1e-3)

    with Plox(style) as px:
        trip_distance.hist(ax=px.a, label=f"{table_name}", color=name2color(table_name))
        px.a.set_yscale('log')
        px.a.set_xlabel('Trip distance, km')
        px.a.set_ylabel('Number of trips')
        px.a.set_xlim(0, 52)
        px.a.set_ylim(1, 2e7)
        px.a.grid(False)
        # px.a.legend()

        px.f.savefig(mkdir(out_dir / f"{whatsmyname()}") / f"{table_name}.png")


def trip_trajectories_initial(table_name):
    N = 10000

    # Column names
    lat = ['xa_lat', 'xb_lat']
    lon = ['xa_lon', 'xb_lon']

    sql = F"SELECT {(', '.join(lat + lon))} FROM [{table_name}] ORDER BY RANDOM() LIMIT {N}"
    df = QUERY(sql)

    with Plox(style) as px:
        px.a.tick_params(axis='both', which='both', labelsize='3')

        for (yy, xx) in zip(df[lat].values, df[lon].values):
            px.a.plot(xx, yy, c=name2color(table_name), ls='-', alpha=0.1, lw=0.1)

        px.a.axis("off")

        # Get the background map
        axis = px.a.axis()
        img_map = maps.get_map_by_bbox(maps.ax2mb(*axis))

        px.a.imshow(img_map, extent=axis, interpolation='quadric', zorder=-100)

        px.f.savefig(mkdir(out_dir / f"{whatsmyname()}") / f"{table_name}.png")


def pickup_hour_heatmap(table_name):
    sql = F"SELECT [ta] FROM [{table_name}]"
    # sql += "ORDER BY RANDOM() LIMIT 1000"  # DEBUG

    pickup = pd.to_datetime(QUERY(sql).squeeze())
    assert isinstance(pickup, pd.Series)

    # Number of rides by weekday and hour
    df: pd.DataFrame
    df = pd.DataFrame({'d': pickup.dt.weekday, 'h': pickup.dt.hour})
    df = df.groupby(['d', 'h']).size().reset_index()
    df = df.pivot(index='d', columns='h', values=0)
    df = df.sort_index()

    # Average over the number of weekdays in the dataset
    df = df.div(pd.Series(Counter(d for (d, g) in groupby(pickup.sort_values().dt.weekday))), axis='index')

    with Plox(style) as px:
        im = px.a.imshow(df, cmap="Blues", origin="upper")

        (xlim, ylim) = (px.a.get_xlim(), px.a.get_ylim())
        px.a.set_xticks(np.linspace(-0.5, 23.5, 25))
        px.a.set_xticklabels(range(0, 25), fontsize="xx-small")
        px.a.set_yticks(px.a.get_yticks(minor=False), minor=False)
        px.a.set_yticklabels([dict(enumerate(calendar.day_abbr)).get(int(t), "") for t in px.a.get_yticks(minor=False)])
        px.a.set_xlim(*xlim)
        px.a.set_ylim(*ylim)

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                alignment = dict(ha="center", va="center")
                im.axes.text(j, i, int(round(df.loc[i, j])), fontsize=4, **alignment)

        px.f.savefig(mkdir(out_dir / f"{whatsmyname()}") / f"{table_name}.png")


def trip_speed_histogram(table_name):
    import matplotlib.pyplot as plt

    sql = F"SELECT [ta] as ta, [tb] as tb, [distance] as m FROM [{table_name}]"
    # sql += "ORDER BY RANDOM() LIMIT 1000"  # DEBUG
    df = QUERY(sql)
    df.ta = pd.to_datetime(df.ta)
    df.tb = pd.to_datetime(df.tb)
    # Duration in seconds
    df['s'] = (df.tb - df.ta).dt.total_seconds()
    # Hour of the day
    df['h'] = df.tb.dt.hour
    # Forget pickup/dropoff times
    df = df.drop(columns=['ta', 'tb'])
    # Columns: distance-meters, duration-seconds, hour-of-the-day
    assert (all(df.columns == ['m', 's', 'h']))

    # Estimated average trip speed
    df['v'] = df.m / df.s

    (H, Min, Sec) = (1 / (60 * 60), 1 / 60, 1)

    # Omit low- and high-speed trips [m/s]
    MIN_TRIP_SPEED = 0.1  # m/s
    MAX_TRIP_SPEED = 15.5  # m/s
    df = df[df.v >= MIN_TRIP_SPEED]
    df = df[df.v <= MAX_TRIP_SPEED]

    # Omit long-duration trips
    MAX_TRIP_DURATION_H = 2  # hours
    df = df[df.s < MAX_TRIP_DURATION_H * (Sec / H)]

    with plt.style.context({**Plox._default_style, **style}):
        fig: plt.Figure
        ax1: plt.Axes
        # Note: the default figsize is W x H = 8 x 6
        (fig, ax24) = plt.subplots(24, 1, figsize=(8, 12))

        xticks = list(range(int(np.ceil(MAX_TRIP_SPEED))))

        for (h, hdf) in df.groupby(df['h']):
            c = name2color(table_name) or plt.get_cmap("twilight_shifted")([h / 24])

            ax1 = ax24[h]
            ax1.hist(hdf.v, bins='fd', lw=2, density=True, histtype='step', color=c, zorder=10)

            ax1.set_xlim(0, MAX_TRIP_SPEED)
            ax1.set_xticks(xticks)
            ax1.set_xticklabels([])

            ax1.set_yticks([])
            ax1.set_ylabel(F"{h}h", fontsize=10, rotation=90)
            ax1.grid()

        ax1.set_xticklabels(xticks)
        ax1.set_xlabel("m/s")

        fig.savefig(mkdir(out_dir / f"{whatsmyname()}") / f"{table_name}.png")


def running_number_of_trips(table_name):
    sql = F"""
        SELECT [ta] as t, +1 as n FROM [{table_name}]
        UNION ALL
        SELECT [tb] as t, -1 as n FROM [{table_name}]
    """

    df: pd.DataFrame
    df = QUERY(sql)
    df.t = pd.to_datetime(df.t)
    df = df.sort_values(by='t')
    df.n = np.cumsum(df.n)

    df['d'] = df.t.dt.floor('1d')
    df['h'] = df.t.dt.hour

    df = df.groupby(['d', 'h']).mean().reset_index()
    df = df.pivot(index='d', columns='h', values='n').fillna(0).sort_index()

    with Plox(style) as px:
        im = px.a.imshow(df, cmap=name2cmap(table_name), origin="upper")

        (xlim, ylim) = (px.a.get_xlim(), px.a.get_ylim())
        px.a.set_xticks(np.linspace(-0.5, 23.5, 25))
        px.a.set_xticklabels(range(25), fontsize=6)
        px.a.set_yticks(range(len(df.index)))
        px.a.set_yticklabels(df.index.date, fontsize=5)
        px.a.set_xlim(*xlim)
        px.a.set_ylim(*ylim)

        for (j, c) in enumerate(df.columns):
            for (i, x) in enumerate(df[c]):
                im.axes.text(j, i, int(x), ha="center", va="center", fontsize=3)

        px.f.savefig(mkdir(out_dir / f"{whatsmyname()}") / f"{table_name}.png")


def trip_duration_vs_distance(table_name):
    N = 100000

    sql = f"""
        SELECT [ta], [tb], [distance] as ds
        FROM [{table_name}]
        ORDER BY RANDOM() 
        LIMIT {N}
    """

    # WHERE ('2016-05-01 00:00' <= ta) and (tb <= '2016-05-02 08:30')

    df = QUERY(sql)
    df['dt'] = pd.to_datetime(df.tb) - pd.to_datetime(df.ta)

    with Plox(style) as px:
        px.a.scatter(df['ds'] / 1e3, df['dt'].dt.total_seconds() / 60, alpha=0.01, edgecolors='none', c=name2color(table_name), zorder=N)
        px.a.set_xlabel("Distance, km")
        px.a.set_ylabel("Time, min")
        px.a.set_xlim(0, 13)
        px.a.set_ylim(0, 33)
        px.a.grid(True, zorder=0)
        px.f.savefig(mkdir(out_dir / f"{whatsmyname()}") / f"{table_name}.png")


if __name__ == '__main__':
    ff = [
        trip_distance_histogram,
        trip_trajectories_initial,
        pickup_hour_heatmap,
        trip_speed_histogram,
        running_number_of_trips,
        trip_duration_vs_distance,
    ]

    tables = {"green_tripdata_2016-05", "yellow_tripdata_2016-05"}

    for f in ff:
        for table_name in sorted(tables):
            log.info(f"{f.__name__}({table_name})")
            f(table_name)
