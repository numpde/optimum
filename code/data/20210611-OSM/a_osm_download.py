# RA, 2019-10-21
# RA, 2021-06-11

import shutil
import functools
import requests
from pathlib import Path
from tcga.utils import unlist1, mkdir, relpath
from twig import log
from zipfile import ZipFile, ZIP_DEFLATED

BASE = Path(__file__).parent

PARAM = {
    'api_url': "https://overpass-api.de/api/interpreter",
    # Note: check http://overpass-api.de/api/status

    'queries': {
        'manhattan': unlist1(BASE.parent.glob("**/manhattan.overpassql")).open(mode='r').read(),
    },

    'zipped_name': "data.json",
}


def download_osm(to_file: Path, overpass_query: str):
    if to_file.is_file():
        log.info(f"File {relpath(to_file)} already exists -- skipping download.")
        return

    with requests.post(PARAM['api_url'], {'data': overpass_query}, stream=True) as response:
        if (response.status_code != 200):
            log.warning(f"Overpass status code: {response.status_code} -- download aborted.")
            return

        # https://github.com/psf/requests/issues/2155#issuecomment-50771010
        response.raw.read = functools.partial(response.raw.read, decode_content=True)

        with ZipFile(to_file, mode='w', compression=ZIP_DEFLATED, compresslevel=9) as zf:
            with zf.open(PARAM['zipped_name'], mode='w', force_zip64=True) as fd:
                shutil.copyfileobj(response.raw, fd)


if __name__ == "__main__":
    for (place, query) in PARAM['queries'].items():
        log.info(f"Downloading `{place}`.")
        download_osm(to_file=(mkdir((BASE / "a_osm") / place) / "osm_json.zip"), overpass_query=query)
