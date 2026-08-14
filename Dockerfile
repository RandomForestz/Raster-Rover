FROM python:3.12-slim

WORKDIR /app

# rasterio/geopandas/pyogrio bundle almost all of GDAL's own dependencies
# inside the wheel (libtiff, libproj, libgeos, etc. all ship in
# rasterio.libs/ and pyogrio.libs/), but GDAL still dynamically links a
# handful of very common libraries from the base OS rather than vendoring
# them. python:3.12-slim doesn't include these by default, so without this
# step you'd hit "cannot open shared object file" errors (libexpat is the
# one that shows up first, but the others are needed too).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    libgcc-s1 \
    libstdc++6 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY raster_rover.py wsgi.py ./

# Render sets $PORT; default to 8080 for local `docker run`.
ENV PORT=8080
EXPOSE 8080

CMD gunicorn -w 2 -b 0.0.0.0:$PORT wsgi:app --timeout 120
