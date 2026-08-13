FROM python:3.12-slim

WORKDIR /app

# rasterio/geopandas/pyogrio ship their own bundled GDAL inside the wheel,
# so no system gdal-bin/libgdal-dev is needed here.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY raster_rover.py wsgi.py ./

# Render sets $PORT; default to 8080 for local `docker run`.
ENV PORT=8080
EXPOSE 8080

CMD gunicorn -w 2 -b 0.0.0.0:$PORT wsgi:app --timeout 120
