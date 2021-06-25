# RA, 2021-06-11

"""
Based on
https://github.com/numpde/transport/blob/master/pt2pt/20191021-NYCTLC/data_preparation/b_explore_taxidata.py
"""

import calendar
import contextlib
import json
from collections import Counter
from datetime import timedelta

from scipy.stats import gaussian_kde
from twig import log

from pathlib import Path
from sqlite3 import connect

import numpy as np
import pandas as pd

from itertools import groupby

from geopy.distance import distance as geodistance
from tqdm import tqdm as progress
import percache

from plox import Plox, rcParam
from tcga.utils import unlist1, mkdir, whatsmyname

from opt_maps import maps
from opt_utils.misc import Section
from opt_utils.style import default_style, name2color, name2cmap

BASE = Path(__file__).parent.resolve()
out_dir = Path(__file__).with_suffix('')

cache = percache.Cache(str(mkdir(Path(__file__).with_suffix('')) / "percache.dat"), livesync=True)


def QUERY(sql) -> pd.DataFrame:
    log.debug(f"Query: {sql}.")
    with connect(unlist1(BASE.glob("**/trips.db"))) as con:
        return pd.read_sql_query(sql, con, parse_dates=['ta', 'tb'])


def trip_table(table_name):
    sql = f"SELECT ta as pickup, tb as dropoff, xa_lat, xa_lon, xb_lat, xb_lon, passenger_count as n, distance as meters, duration as seconds, total_amount as [$] FROM [{table_name}] ORDER BY pickup LIMIT 7"
    df = QUERY(sql)
    df.meters = np.round(df.meters).astype(int)
    df.seconds = np.round(df.seconds).astype(int)
    df.iloc[-1, 1:] = ""
    df.iloc[-1, 0] = "..."
    with (mkdir(out_dir / f"{whatsmyname()}") / f"{table_name}.tex").open(mode='w') as fd:
        print(df.to_latex(index=False, na_rep=""), file=fd)


def trip_distance_histogram(table_name):
    sql = f"SELECT [distance] FROM [{table_name}]"
    trip_distance = QUERY(sql).squeeze()

    assert isinstance(trip_distance, pd.Series)

    # Convert to km
    trip_distance *= (1e-3)

    with Plox({**default_style, rcParam.Figure.figsize: (8, 3)}) as px:
        trip_distance.hist(ax=px.a, label=f"{table_name}", color=name2color(table_name), edgecolor="white")
        px.a.set_yscale('log')
        px.a.set_xlabel('Trip distance, km')
        px.a.set_ylabel('Number of trips')
        px.a.set_xlim(0, 52)
        px.a.set_ylim(1, 2e7)
        px.a.grid(linewidth=0.2)
        # px.a.legend()

        px.f.savefig(mkdir(out_dir / f"{whatsmyname()}") / f"{table_name}.png")


def trip_trajectories_initial(table_name):
    N = 10000

    # Column names
    lat = ['xa_lat', 'xb_lat']
    lon = ['xa_lon', 'xb_lon']

    sql = f"SELECT {(', '.join(lat + lon))} FROM [{table_name}] ORDER BY RANDOM() LIMIT {N}"
    df = QUERY(sql)

    with Plox(default_style) as px:
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
    sql = f"SELECT [ta] FROM [{table_name}]"
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

    with Plox(default_style) as px:
        im = px.a.imshow(df, cmap=name2cmap(table_name), origin="upper")

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


def trip_fare_vs_distance(table_name):
    limit = 111111

    sql = f"""
        SELECT [ta], [tb], [total_amount] as fare, [distance] as ds, strftime('%w', ta) as w, strftime('%H', ta) as h
        FROM [{table_name}]
        WHERE (fare > 0) and ((100 < distance) and (distance < 2000))
        ORDER BY RANDOM() 
        LIMIT {limit}
    """

    trips = QUERY(sql).astype({'h': int})

    trips = trips[trips.fare <= np.quantile(trips.fare, 0.99)]

    from matplotlib.colors import LinearSegmentedColormap as LinearColormap
    from matplotlib.ticker import MaxNLocator

    with Plox(default_style) as px:
        px.a.grid(lw=0.3)
        for (h, df) in trips.groupby(by='h'):
            trip_dist = df.ds
            trip_fare = df.fare

            cmap = LinearColormap.from_list(name="sun", colors=["black", "lightblue", "yellow", "orange", "black"])

            px.a.scatter(
                trip_dist, trip_fare,
                c=cmap([h / 24]), s=3, alpha=0.2, lw=0, zorder=-10,
                label=f"{len(df)} trips at {h}h"
            )

            px.a.set_xlabel(f"Trip distance, m")
            px.a.set_ylabel(f"Trip fare, \$")

            [h.set_alpha(1) for h in px.a.legend(fontsize=7, loc='upper right').legendHandles]

        px.f.savefig(mkdir(out_dir / f"{whatsmyname()}") / f"{table_name}.png")


@cache
def zoom(table_name: str, weekday: int, limit, radius, focus=(40.75798, -73.98550)):
    sql = f"""
        SELECT [ta], [tb], [xa_lat], [xa_lon], [xb_lat], [xb_lon], [distance] as ds, strftime('%w', ta) as weekday
        FROM [{table_name}]
        WHERE (weekday == '{weekday}')
        ORDER BY RANDOM() 
        LIMIT {limit}
    """

    # Note: could do `where` by lat/lon bbox

    trips = QUERY(sql)

    trips = trips.assign(xa=list(zip(trips.xa_lat, trips.xa_lon)))
    trips = trips.assign(xb=list(zip(trips.xb_lat, trips.xb_lon)))

    dist_to_focus = pd.Series(index=trips.index, data=[
        max(geodistance(row.xa, focus).m, geodistance(row.xb, focus).m)
        for (i, row) in progress(trips.iterrows(), total=len(trips))
    ])

    return trips[dist_to_focus <= radius]


def trip_speeds_times_square(table_name):
    query_params = dict(weekday=3, limit=1000000, radius=1000)

    trips = zoom(table_name, **query_params)

    MAX_TRIP_DURATION_Min = 30
    trips = trips.assign(s=(trips.tb - trips.ta).dt.total_seconds())
    trips = trips[trips.s <= timedelta(minutes=MAX_TRIP_DURATION_Min).total_seconds()]

    trips = trips.assign(v=(trips.ds / trips.s))
    (MIN_TRIP_SPEED, MAX_TRIP_SPEED) = (0.1, 15.5)
    trips = trips[(MIN_TRIP_SPEED <= trips.v) & (trips.v <= MAX_TRIP_SPEED)]  # reasonable average speeds

    log.info(f"Got {len(trips)} trips about Times Square from {table_name}.")

    with (mkdir(out_dir / f"{whatsmyname()}") / f"{table_name}.json").open(mode='w') as fd:
        with contextlib.redirect_stdout(fd):
            print(json.dumps(indent=2, obj={
                'query': query_params, 'trips': len(trips),
                'min_trip_speed': MIN_TRIP_SPEED, 'max_trip_speed': MAX_TRIP_SPEED,
                'max_trip_duration_minutes': MAX_TRIP_DURATION_Min,
            }))

    if not len(trips):
        return

    trips = trips.assign(h=trips.ta.dt.hour)  # hour of the day

    from matplotlib.colors import LinearSegmentedColormap as LinearColormap
    from matplotlib.ticker import MaxNLocator

    cmap = LinearColormap.from_list(name="sun", colors=["black", "lightblue", "yellow", "orange", "black"])

    with Plox({**default_style, rcParam.Figure.figsize: (8, 3)}) as px:
        for (h, df) in trips.groupby(by='h'):
            kde = gaussian_kde(df.v)
            xx = np.linspace(0, MAX_TRIP_SPEED, 101)
            yy = kde(xx)

            more = dict(label=f"{h}h") if not (h % 2) else {}
            px.a.plot(xx, yy, lw=(3 if (h in [6, 18]) else 1), c=cmap([h / 24]), alpha=0.7, **more)
            # px.a.hist(df.v, bins='fd', lw=2, density=True, histtype='step', color=cmap([h / 24]), zorder=10, alpha=0.7, label=f"{h}h")

        xticks = list(range(int(np.ceil(MAX_TRIP_SPEED))))
        px.a.set_xlim(0, MAX_TRIP_SPEED)
        px.a.set_xticks(xticks)

        # px.a.set_xscale('log')

        px.a.set_yticks([])
        px.a.grid(lw=0.3)

        px.a.set_xticklabels(xticks)
        px.a.set_xlabel("m/s")

        [h.set_alpha(1) for h in px.a.legend(fontsize=9, loc='upper right').legendHandles]

        px.f.savefig(mkdir(out_dir / f"{whatsmyname()}") / f"{table_name}.png")


def trip_speed_histogram(table_name):
    import matplotlib.pyplot as plt

    sql = f"SELECT [ta] as ta, [tb] as tb, [distance] as m FROM [{table_name}]"
    # sql += "ORDER BY RANDOM() LIMIT 1000"  # DEBUG
    df = QUERY(sql)
    # Duration in seconds
    df['s'] = (df.tb - df.ta).dt.total_seconds()
    # Hour of the day
    df['h'] = df.tb.dt.hour  # why tb?
    # Forget pickup/dropoff times
    df = df.drop(columns=['ta', 'tb'])
    # Columns: distance-meters, duration-seconds, hour-of-the-day
    assert (all(df.columns == ['m', 's', 'h']))

    # Estimated average trip speed
    df['v'] = df.m / df.s

    (H, Min, Sec) = (1 / (60 * 60), 1 / 60, 1)

    # Omit low- and high-speed trips [m/s]
    MIN_TRIP_SPEED = 0.1  # m/s
    MAX_TRIP_SPEED = 15.5  # m/s  (do .5 for plotting)
    df = df[df.v >= MIN_TRIP_SPEED]
    df = df[df.v <= MAX_TRIP_SPEED]

    # Omit long-duration trips
    MAX_TRIP_DURATION_H = 1  # hours
    df = df[df.s < MAX_TRIP_DURATION_H * (Sec / H)]

    with plt.style.context({**Plox._default_style, **default_style}):
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
            ax1.set_ylabel(f"{h}h", fontsize=10, rotation=90)
            ax1.grid()

        ax1.set_xticklabels(xticks)
        ax1.set_xlabel("m/s")

        fig.savefig(mkdir(out_dir / f"{whatsmyname()}") / f"{table_name}.png")


def running_number_of_trips(table_name):
    sql = f"""
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

    with Plox(default_style) as px:
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

    with Plox(default_style) as px:
        traveldist = df['ds'] / 1e3
        traveltime = df['dt'].dt.total_seconds() / 60
        px.a.scatter(traveldist, traveltime, alpha=0.01, edgecolors='none', c=name2color(table_name), zorder=N)
        px.a.set_xlabel("Distance, km")
        px.a.set_ylabel("Time, min")
        px.a.set_xlim(0, 13)
        px.a.set_ylim(0, 33)
        px.a.grid(True, zorder=0)
        px.f.savefig(mkdir(out_dir / f"{whatsmyname()}") / f"{table_name}.png")


def trip_duration_vs_distance2(table_name):
    N = 100000

    sql = f"""
        SELECT [ta], [tb], [distance] as ds
        FROM [{table_name}]
        ORDER BY RANDOM() 
        LIMIT {N}
    """

    # WHERE ('2016-05-01 00:00' <= ta) and (tb <= '2016-05-02 08:30')

    trips = QUERY(sql)

    trips['dt'] = trips.tb - trips.ta

    # Hour of the day
    trips['h'] = trips['ta'].dt.hour

    from matplotlib.colors import LinearSegmentedColormap as LinearColormap
    from matplotlib.ticker import MaxNLocator

    with Plox(default_style) as px:
        px.a.grid()
        # px.a.plot(*(2 * [[0, df[['reported', 'shortest']].values.max()]]), c='k', ls='--', lw=0.3, zorder=100)

        for (h, df) in trips.groupby(by='h'):
            trip_dist = df['ds'] / 1e3
            trip_time = df['dt'].dt.total_seconds() / 60

            cmap = LinearColormap.from_list(name="sun", colors=["black", "lightblue", "yellow", "orange", "black"])
            px.a.scatter(
                trip_time, trip_dist,
                c=cmap([h / 24]), s=3, alpha=0.2, lw=0, zorder=-10,
                label=f"{len(df)} trips at {h}h"
            )

        # px.a.scatter(traveldist, traveltime, alpha=0.01, edgecolors='none', c=name2color(table_name), zorder=N)

        px.a.set_xlabel("Time, min")
        px.a.set_ylabel("Distance, km")
        px.a.set_xlim(0, 33)
        px.a.set_ylim(0, 13)
        px.a.grid(True, zorder=0, lw=0.3)
        [h.set_alpha(1) for h in px.a.legend(fontsize=6, loc='upper left').legendHandles]

        px.a.xaxis.set_major_locator(MaxNLocator(integer=True))
        px.a.yaxis.set_major_locator(MaxNLocator(integer=True))

        px.f.savefig(mkdir(out_dir / f"{whatsmyname()}") / f"{table_name}.png")


if __name__ == '__main__':
    ff = [
        trip_table,
        # trip_fare_vs_distance,
        # trip_speeds_times_square,
        # trip_duration_vs_distance2,
        # trip_distance_histogram,
        # trip_trajectories_initial,
        # pickup_hour_heatmap,
        # trip_speed_histogram,
        # running_number_of_trips,
        # trip_duration_vs_distance,
    ]

    tables = {"green_tripdata_2016-05", "yellow_tripdata_2016-05"}

    for f in ff:
        for table_name in sorted(tables):
            with Section(f"Processing {f.__name__}({table_name})", out=log.info):
                f(table_name)
