# RA, 2018-11-07
# RA, 2019-10-24
# RA, 2021-06-11

import io
import os

import numpy as np

import urllib.request

WGET_TIMEOUT = 20  # In seconds

from enum import Enum
from PIL import Image
from math import pi, log, tan, exp, atan, log2, floor

from urllib.error import URLError
from socket import timeout as TimeoutError

from retry import retry

from pathlib import Path
from tcga.utils import mkdir

import percache

import dotenv

# Load the MapBox token, if present
dotenv.load_dotenv(Path(__file__).with_suffix('.env'))

cache = percache.Cache(str(mkdir(Path(__file__).parent / "cache") / "percache_maps"), livesync=False)

PARAM = {
    'do_retina': True,
    'do_snap_to_dyadic': True,
}

# Convert geographical coordinates to pixels
# https://en.wikipedia.org/wiki/Web_Mercator_projection
#
# Note on google API:
# The world map is obtained with lat=lon=0, w=h=256, zoom=0
# Note on mapbox API:
# The world map is obtained with lat=lon=0, w=h=512, zoom=0
#
# Therefore:
MAPBOX_ZOOM0_SIZE = 512  # Not 256


# https://www.mapbox.com/api-documentation/#styles
# https://docs.mapbox.com/api/maps/styles/
class MapBoxStyle(Enum):
    streets = 'streets-v11'
    outdoors = 'outdoors-v11'
    light = 'light-v10'
    dark = 'dark-v10'
    satellite = 'satellite-v9'
    satellite_streets = 'satellite-streets-v11'
    navi_preview_day = 'navigation-preview-day-v4'
    navi_preview_night = 'navigation-preview-night-v4'
    navi_guide_day = 'navigation-guidance-day-v4'
    navi_guide_night = 'navigation-guidance-night-v4'


# Geo-coordinate in degrees => Pixel coordinate
def g2p(lat, lon, zoom):
    return (
        # x
        MAPBOX_ZOOM0_SIZE * (2 ** zoom) * (1 + lon / 180) / 2,
        # y
        MAPBOX_ZOOM0_SIZE / (2 * pi) * (2 ** zoom) * (pi - log(tan(pi / 4 * (1 + lat / 90))))
    )


# Pixel coordinate => geo-coordinate in degrees
def p2g(x, y, zoom):
    return (
        # lat
        (atan(exp(pi - y / MAPBOX_ZOOM0_SIZE * (2 * pi) / (2 ** zoom))) / pi * 4 - 1) * 90,
        # lon
        (x / MAPBOX_ZOOM0_SIZE * 2 / (2 ** zoom) - 1) * 180,
    )


def ax2mb(left, right, bottom, top):
    return (left, bottom, right, top)


def mb2ax(left, bottom, right, top):
    return (left, right, bottom, top)


def ax4(lats, lons, extra_space=0.01):
    extent = np.dot(
        [
            [min(lons), max(lons)],
            [min(lats), max(lats)],
        ],
        (
            lambda s:
            np.asarray([[1 + s, -s], [-s, 1 + s]])
        )(extra_space)
    )
    return tuple(extent.flatten())


def mb4(lat, lon):
    (left, bottom, right, top) = [min(lon), min(lat), max(lon), max(lat)]
    return (left, bottom, right, top)


@cache
@retry((URLError, TimeoutError), tries=3, delay=1)
def wget(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=WGET_TIMEOUT) as response:
        return response.read()


# bbox = (left, bottom, right, top) in degrees
def get_map_by_bbox(bbox, token=os.getenv("MAPBOX_TOKEN"), style=MapBoxStyle.light):
    if not token:
        raise RuntimeError("An API token is required")

    # The region of interest in geo-coordinates in degrees
    (left, bottom, right, top) = bbox
    # Sanity check
    assert (-90 <= bottom < top <= 90)
    assert (-180 <= left < right <= 180)

    # Rendered image map size in pixels as it should come from MapBox (no retina)
    (w, h) = (1024, 1024)

    # The center point of the region of interest
    (lat, lon) = ((top + bottom) / 2, (left + right) / 2)

    # Reduce precision of (lat, lon) to increase cache hits
    if PARAM['do_snap_to_dyadic']:
        snap_to_dyadic = (lambda a, b: (lambda x, scale=(2 ** floor(log2(abs(b - a) / 4))): (round(x / scale) * scale)))
        lat = snap_to_dyadic(bottom, top)(lat)
        lon = snap_to_dyadic(left, right)(lon)

        assert ((bottom < lat < top) and (left < lon < right)), "Reference point not inside the region of interest"

    # Look for appropriate zoom level to cover the region of interest by that map
    for zoom in range(16, 0, -1):
        # Center point in pixel coordinates at this zoom level
        (x0, y0) = g2p(lat, lon, zoom)
        # The geo-region that the downloaded map would cover
        ((TOP, LEFT), (BOTTOM, RIGHT)) = (p2g(x0 - w / 2, y0 - h / 2, zoom), p2g(x0 + w / 2, y0 + h / 2, zoom))
        # Would the map cover the region of interest?
        if (LEFT <= left < right <= RIGHT) and (BOTTOM <= bottom < top <= TOP):
            break

    # Choose "retina" quality of the map
    retina = {True: "@2x", False: ""}[PARAM['do_retina']]

    # Assemble the query URL
    url = F"https://api.mapbox.com/styles/v1/mapbox/{style.value}/static/{lon},{lat},{zoom}/{w}x{h}{retina}?access_token={token}&attribution=false&logo=false"

    # Download the rendered image
    b = wget(url)

    # Convert bytes to image object
    I = Image.open(io.BytesIO(b), mode='r')

    # # DEBUG: show image
    # import matplotlib as mpl
    # mpl.use("TkAgg")
    # import matplotlib.pyplot as plt
    # plt.imshow(I)
    # plt.show()
    # exit(39)

    # If the "retina" @2x parameter is used, the image is twice the size of the requested dimensions
    (W, H) = I.size
    assert ((W, H) in [(w, h), (2 * w, 2 * h)])

    # Extract the region of interest from the larger covering map
    i = I.crop((
        round(W * (left - LEFT) / (RIGHT - LEFT)),
        round(H * (top - TOP) / (BOTTOM - TOP)),
        round(W * (right - LEFT) / (RIGHT - LEFT)),
        round(H * (bottom - TOP) / (BOTTOM - TOP)),
    ))

    return i


def test():
    bbox = [120.2206, 22.4827, 120.4308, 22.7578]
    map = get_map_by_bbox(bbox)

    import matplotlib as mpl
    mpl.use("TkAgg")

    import matplotlib.pyplot as plt
    plt.imshow(map, extent=mb2ax(*bbox))
    plt.show()


if __name__ == "__main__":
    test()
