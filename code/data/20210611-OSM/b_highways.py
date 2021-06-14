# RA, 2019-10-26
# RA, 2021-06-11

from twig import log

import pandas as pd
import numpy as np

from itertools import chain

import json

from zipfile import ZipFile
from pathlib import Path
from collections import Counter

from tcga.utils import unlist1, mkdir
from plox import Plox

from opt_maps import maps
from opt_utils.style import default_style

BASE = Path(__file__).parent
out_dir = mkdir(BASE.with_suffix(''))

HW_KEY = "highway"


def extract_highways(j):
    log.info(f"Element summary: {dict(Counter(x['type'] for x in j))}.")

    class highways:
        nodes = pd.DataFrame(data=[x for x in j if (x['type'] == "node")]).set_index('id', verify_integrity=True)
        ways = pd.DataFrame(data=[x for x in j if (x['type'] == "way")]).set_index('id', verify_integrity=True)

        log.info(f"Highway summary: {dict(Counter(tag.get(HW_KEY) for tag in ways['tags']))}.")

        # All OSM ways tagged "highway"
        ways[HW_KEY] = [tags.get(HW_KEY) for tags in ways['tags']]
        ways = ways[~ways[HW_KEY].isna()]
        ways = ways.drop(columns=['type'])

        # Retain only nodes that support any remaining ways
        nodes = nodes.loc[set(chain.from_iterable(ways.nodes.values)), :]

    return highways


def propose_highway_matrix(highways):
    # Preallocate
    highway_matrix = pd.DataFrame(
        index=pd.Index(highways.ways[HW_KEY].unique(), name=HW_KEY),
        columns=["drivable", "cyclable", "walkable"],
    )

    highway_matrix['drivable'][{
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
        'unclassified', 'residential', 'living_street',
    }] = True

    highway_matrix['drivable'][
        {'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'}
    ] = True

    highway_matrix['drivable'][{'service'}] = True

    # highway_matrix['drivable'][{'pedestrian'}] = True
    #
    highway_matrix['cyclable'][{'cycleway'}] = True
    #
    highway_matrix['walkable'][
        {'footway', 'pedestrian', 'living_street', 'path', 'elevator', 'platform', 'steps', 'corridor'}
    ] = True

    return highway_matrix


def plot_highways(highways):
    highway_matrix = propose_highway_matrix(highways)

    for highway_kind in highway_matrix.columns:
        log.info(highway_kind)

        # Sub-dataframe of OSM ways
        ways0 = highways.ways[(True == highway_matrix[highway_kind][highways.ways[HW_KEY]]).values]

        # print(drivable_ways.groupby(drivable_ways[HW].apply(lambda i: i.split('_')[0])).size().sort_values(ascending=False).index)
        # print(pd.Series(index=(drivable_ways[HW].apply(lambda i: i.split('_')[0]))).groupby(level=0).size().sort_values(ascending=False).index)
        # print(list(reversed(drivable_ways.groupby(HW).size().groupby(lambda i: i.split('_')[0]).sum().sort_values().index)))
        # print(list(drivable_ways.groupby(HW).size().groupby(lambda i: i.split('_')[0]).sum().sort_values(ascending=False).index))
        # exit(9)

        highway_labels = list(
            ways0.groupby(HW_KEY).size().groupby(lambda i: i.split('_')[0]).sum().sort_values(ascending=False).index
        )

        with Plox(default_style) as px:
            px.a.tick_params(axis='both', which='both', labelsize=4)

            extent = np.dot(
                [[min(highways.nodes.lon), max(highways.nodes.lon)],
                 [min(highways.nodes.lat), max(highways.nodes.lat)]],
                (lambda s: np.asarray([[1 + s, -s], [-s, 1 + s]]))(0.01)
            ).flatten()

            px.a.set_xlim(extent[0:2])
            px.a.set_ylim(extent[2:4])

            for (n, label) in enumerate(highway_labels):
                for way in ways0[ways0[HW_KEY] == label]['nodes']:
                    (lat, lon) = highways.nodes.loc[way, ['lat', 'lon']].values.T
                    px.a.plot(lon, lat, '-', c=(f"C{n}"), alpha=0.9, lw=0.2, label=label)
                    px.a.set_xlabel("longitude", fontdict=dict(size="xx-small"))
                    px.a.set_ylabel("latitude", fontdict=dict(size="xx-small"))
                    # Avoid thousands of legend entries:
                    label = None

            px.a.legend(loc="upper left", fontsize="xx-small")

            # Get the background map
            px.a.imshow(maps.get_map_by_bbox(maps.ax2mb(*extent)), extent=extent, interpolation='quadric', zorder=-100)

            yield (highway_kind, px)


def main():
    for area in BASE.glob("a_osm/*/osm_json.zip"):
        with ZipFile(area, mode='r') as zf:
            with zf.open(unlist1(zf.namelist()), mode='r') as fd:
                highways = extract_highways((json.load(fd))['elements'])

                propose_highway_matrix(highways).to_csv(
                    mkdir(out_dir / "highway_matrix") / f"{area.parent.name}.tsv", sep='\t'
                )

                for (kind, px) in plot_highways(highways):
                    px.f.savefig(mkdir(out_dir / f"figs/{area.parent.name}") / f"{kind}.png")


if __name__ == '__main__':
    main()
