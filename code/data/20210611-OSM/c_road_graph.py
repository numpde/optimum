# RA, 2019-10-26
# RA, 2021-06-11

from twig import log

import json
import pickle

import pandas as pd
import numpy as np
import networkx as nx

from geopy.distance import geodesic as distance

from itertools import chain
from pathlib import Path
from more_itertools import pairwise

from zipfile import ZipFile

from opt_maps import maps
from opt_utils.misc import Section
from opt_utils.style import default_style
import opt_utils.graph as graph

from plox import Plox
from tcga.utils import unlist1, mkdir

BASE = Path(__file__).parent
out_dir = mkdir(Path(__file__).with_suffix(''))

PARAM = {
    'way_tags_we_like': [
        "name", "highway", "sidewalk", "private", "bus",
        "cycleway", "oneway", "foot", "pedestrian", "turn",
    ],

    'max_graph_edge_len': 20,
}


def process(osm_json: Path):
    area = osm_json.parent.name  # e.g. 'manhattan'
    hw_matrix = pd.read_csv(unlist1(BASE.glob(f"**/highway*/{area}.tsv")), sep='\t', index_col=0)

    with Section("Loading OSM archive", out=log.info):
        with ZipFile(osm_json, mode='r') as zf:
            J = json.load(zf.open(unlist1(zf.namelist())))['elements']

        # OSM nodes and OSM ways as DataFrame
        nodes: pd.DataFramex
        ways: pd.DataFrame
        (nodes, ways) = [
            pd.DataFrame(
                data=(x for x in J if (x['type'] == t))
            ).set_index('id', verify_integrity=True).drop(columns=['type'])
            for t in ["node", "way"]
        ]

    with Section("Filtering", out=log.info):
        # Keep only useful tags
        assert ("oneway" in PARAM['way_tags_we_like'])
        ways.tags = [{k: v for (k, v) in tags.items() if (k in PARAM['way_tags_we_like'])} for tags in ways.tags]

        # Restrict to drivable OSM ways
        drivable = hw_matrix['drivable'].fillna(False)
        ways = ways.loc[(drivable[tags.get('highway')] for tags in ways['tags']), :]

        # Retain only nodes that support any remaining ways
        nodes = nodes.loc[set(chain.from_iterable(ways.nodes.values)), :]

    with Section("Making the graph", out=log.info):
        nodes['loc'] = list(zip(nodes.lat, nodes.lon))
        nodes['pos'] = list(zip(nodes.lon, nodes.lat))

        G = nx.DiGraph()

        for (osm_id, way) in ways.iterrows():
            G.add_edges_from(pairwise(way['nodes']), osm_id=osm_id, **way['tags'])
            if not ("yes" == str.lower(way['tags'].get('oneway', "no"))):
                # https://wiki.openstreetmap.org/wiki/Key:oneway
                G.add_edges_from(pairwise(reversed(way['nodes'])), osm_id=osm_id, **way['tags'])

        def edge_len(uv):
            return (uv, distance(nodes['loc'][uv[0]], nodes['loc'][uv[1]]).m)

        nx.set_edge_attributes(G, name="len", values=dict(map(edge_len, G.edges)))
        nx.set_node_attributes(G, name="loc", values=dict(nodes['loc']))
        nx.set_node_attributes(G, name="pos", values=dict(nodes['pos']))

    with Section("Breaking down long edges", out=log.info):
        log.info(F"Before: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges.")
        graph.break_long_edges(G, max_edge_len=PARAM['max_graph_edge_len'])
        log.info(F"After:  {G.number_of_nodes()} nodes / {G.number_of_edges()} edges.")

        # Update node positions
        nodes = pd.DataFrame(data=nx.get_node_attributes(G, name="loc"), index=["lat", "lon"]).T
        nodes['loc'] = list(zip(nodes.lat, nodes.lon))
        nodes['pos'] = list(zip(nodes.lon, nodes.lat))

        # Node position for plotting
        nx.set_node_attributes(G, name="pos", values=dict(nodes['pos']))

    with Section("Saving graph", out=log.info):
        pickle.dump(G, open(out_dir / f"{area}.pkl", 'wb'))

    with Section("Making figure", out=log.info):
        with Plox(default_style) as px:
            px.a.tick_params(axis='both', which='both', labelsize=3)

            extent = np.dot(
                [[min(nodes.lon), max(nodes.lon)], [min(nodes.lat), max(nodes.lat)]],
                (lambda s: np.asarray([[1 + s, -s], [-s, 1 + s]]))(0.01)
            ).flatten()

            nx.draw_networkx(
                G.to_undirected(),
                ax=px.a,
                pos=nx.get_node_attributes(G, "pos"),
                with_labels=False,
                arrows=False,
                node_size=0, alpha=0.9, width=0.3
            )

            px.a.set_xlim(extent[0:2])
            px.a.set_ylim(extent[2:4])

            # Background map
            px.a.imshow(maps.get_map_by_bbox(maps.ax2mb(*extent)), extent=extent, interpolation='quadric', zorder=-100)

            px.f.savefig(out_dir / f"{area}.png")


def main():
    for path_to_json in BASE.glob("**/osm_json.zip"):
        process(path_to_json)


if __name__ == '__main__':
    main()
