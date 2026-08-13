"""
wsgi.py — production entrypoint for Raster Rover.

Used by gunicorn (see Dockerfile: `gunicorn -b 0.0.0.0:$PORT wsgi:app`).
Not used for local development — for that, just run `python3 raster_rover.py`.

Env vars (all optional):
    ROVER_RASTER   Path to a .tif baked into the image to serve by default
                    (omit to fall back to the built-in demo DEM)
    ROVER_BAND     Band number to read (default 1)
    ROVER_MAX_DIM  Max grid dimension in cells (default 220)
    ROVER_CMAP     Default color palette (default "terrain" for the demo,
                    "viridis" otherwise)
"""

import os

from raster_rover import create_app, make_demo_dem, read_raster

RASTER_PATH = os.environ.get("ROVER_RASTER")
BAND = int(os.environ.get("ROVER_BAND", "1"))
MAX_DIM = int(os.environ.get("ROVER_MAX_DIM", "220"))

if RASTER_PATH:
    arr, crs_display, transform, crs_wkt = read_raster(RASTER_PATH, band=BAND, max_dim=MAX_DIM)
    source_name = os.path.basename(RASTER_PATH)
    cmap_name = os.environ.get("ROVER_CMAP", "viridis")
else:
    arr, crs_display, transform, crs_wkt = make_demo_dem()
    source_name = "Demo DEM (built-in example — add your own .tif above)"
    cmap_name = os.environ.get("ROVER_CMAP", "terrain")

app = create_app(arr, transform, crs_wkt, crs_display, source_name, cmap_name, MAX_DIM, BAND)
