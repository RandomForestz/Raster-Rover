# Deploying Raster Rover to Render

## Files involved
- `raster_rover.py` — the app itself (unchanged from local use)
- `wsgi.py` — production entrypoint gunicorn uses to load the Flask app
- `requirements.txt` — pinned dependencies
- `Dockerfile` — builds the container (no system GDAL needed — rasterio,
  geopandas, and pyogrio all ship GDAL bundled inside their wheels)
- `render.yaml` — optional, lets Render auto-configure the service
- `.dockerignore` — keeps test files out of the image

All of this was tested locally: pip-installed the exact `requirements.txt`
into a clean virtualenv, ran the exact gunicorn command the Dockerfile uses,
and confirmed the homepage and a real GPKG export both work.

## Steps

1. **Push these files to a GitHub repo** (all six files above, in the repo root).

2. **On Render.com:**
   - New → Web Service → connect your GitHub repo
   - Render will detect `render.yaml` automatically and set Runtime to Docker.
     If it doesn't, set it manually: Runtime = Docker, Dockerfile Path = `./Dockerfile`
   - Plan: Free (spins down after ~15 min idle; first request after that takes
     ~30–60s to wake back up — normal for free tier, not a bug)
   - Deploy

3. **That's it.** Render builds the image, starts gunicorn, and gives you a
   URL like `https://raster-rover.onrender.com`.

## Optional: bake in your own default raster

By default the deployed app starts with the built-in demo DEM (same as
running it locally with no arguments). If you'd rather it start with your
own raster:

1. Add your `.tif` to the repo (or somewhere the build can fetch it)
2. In the Dockerfile, add a `COPY yourfile.tif ./` line
3. In Render's environment variables, set `ROVER_RASTER=/app/yourfile.tif`

Users can still use the "Add .tif" button to swap in their own file at
any time — this just changes what loads by default.

## Environment variables (all optional)

| Variable | Default | Purpose |
|---|---|---|
| `ROVER_RASTER` | (none — uses demo DEM) | Path to a raster baked into the image |
| `ROVER_BAND` | `1` | Band to read from that raster |
| `ROVER_MAX_DIM` | `220` | Max grid dimension (higher = finer detail, more memory/CPU) |
| `ROVER_CMAP` | `terrain` (demo) / `viridis` (custom) | Default color palette |

## A note on the free tier

Render's free plan gives limited RAM (usually 512MB). The app itself is
light, but every visitor's raster (including uploads) sits in memory for
that whole session — one upload endpoint enforces a 100MB file-size cap for
this reason. If you expect heavy concurrent use, bump to a paid instance
type or lower `ROVER_MAX_DIM` to shrink each session's memory footprint.
