#!/usr/bin/env python3
"""
raster_rover.py — Raster Rover Exploration Viewer

An in-browser tool for exploring a raster and pulling discrete selections
out of it. Two ways to find cells:

  - HOVER: move the mouse over the grid; the cell under it, and every
    other cell within a threshold of its value, pop.
  - SELECT: type a value, pick a mode (within / above / below), and click
    Select — matching cells are highlighted and stay highlighted.

Whatever the current selection criteria are, you can export them as:
  - a GeoPackage of cell polygons or cell-center points
  - a binary GeoTIFF (selected cells = 1, everything else = NoData)
  - a masked GeoTIFF (selected cells keep their value, everything else = NoData)

Starts with a built-in synthetic demo DEM so there's always something to
explore. Use the "Add .tif" button in the browser to load your own
GeoTIFF, or pass a path on the command line.

Usage:
    python3 raster_rover.py                       # opens with the demo DEM
    python3 raster_rover.py path/to/raster.tif     # opens with your raster
    python3 raster_rover.py --band 2 --max-dim 300 --cmap terrain

This runs a small local Flask server (127.0.0.1 by default) because file
uploads and GPKG/GeoTIFF writing both need a real Python process on the
other end — a browser can't do either on its own. Nothing leaves your
machine.
"""

import argparse
import json
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Raster reading
# --------------------------------------------------------------------------

def read_raster(path, band=1, max_dim=220):
    """Read a raster band, mask nodata as NaN, downsample if large.

    Returns: arr, crs_display (str), transform (affine.Affine for the
    returned grid), crs_wkt (str or None).
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        nodata = src.nodata
        h, w = src.height, src.width

        longer = max(h, w)
        factor = max(1, int(np.ceil(longer / max_dim)))

        if factor > 1:
            out_h = max(1, h // factor)
            out_w = max(1, w // factor)
            arr = src.read(
                band, out_shape=(out_h, out_w), resampling=Resampling.average
            ).astype("float64")
            transform = src.transform * src.transform.scale((w / out_w), (h / out_h))
        else:
            arr = src.read(band).astype("float64")
            transform = src.transform

        crs_display = str(src.crs) if src.crs else "none found in source"
        crs_wkt = src.crs.to_wkt() if src.crs else None

    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan

    return arr, crs_display, transform, crs_wkt


def make_demo_dem(size=129, roughness=0.55, max_elev=1800.0, seed=0):
    """Synthetic elevation raster (diamond-square fractal terrain) with an
    arbitrary local 30m transform, no real CRS."""
    from affine import Affine

    rng = np.random.default_rng(seed)
    n = 1
    while (2 ** n) + 1 < size:
        n += 1
    dim = (2 ** n) + 1

    grid = np.zeros((dim, dim))
    grid[0, 0] = rng.uniform(-1, 1)
    grid[0, -1] = rng.uniform(-1, 1)
    grid[-1, 0] = rng.uniform(-1, 1)
    grid[-1, -1] = rng.uniform(-1, 1)

    step = dim - 1
    scale = 1.0
    while step > 1:
        half = step // 2
        for y in range(half, dim - 1, step):
            for x in range(half, dim - 1, step):
                avg = (
                    grid[y - half, x - half] + grid[y - half, x + half] +
                    grid[y + half, x - half] + grid[y + half, x + half]
                ) / 4.0
                grid[y, x] = avg + rng.uniform(-1, 1) * scale
        for y in range(0, dim, half):
            for x in range((y + half) % step, dim, step):
                total, count = 0.0, 0
                if y - half >= 0:
                    total += grid[y - half, x]; count += 1
                if y + half < dim:
                    total += grid[y + half, x]; count += 1
                if x - half >= 0:
                    total += grid[y, x - half]; count += 1
                if x + half < dim:
                    total += grid[y, x + half]; count += 1
                grid[y, x] = total / count + rng.uniform(-1, 1) * scale
        step = half
        scale *= roughness

    grid = grid[:size, :size]
    grid -= grid.min(); grid /= grid.max()

    y, x = np.mgrid[0:size, 0:size]
    bias = np.exp(-((x - size * 0.35) ** 2 + (y - size * 0.65) ** 2) / (2 * (size * 0.28) ** 2))
    grid = 0.75 * grid + 0.35 * bias
    grid -= grid.min(); grid /= grid.max()

    elevation = grid * max_elev
    elevation[int(size * 0.72):int(size * 0.80), int(size * 0.15):int(size * 0.24)] = np.nan

    transform = Affine.translation(0, size * 30) * Affine.scale(30, -30)
    return elevation, "Demo DEM (synthetic, meters, no real CRS)", transform, None


# --------------------------------------------------------------------------
# Coloring
# --------------------------------------------------------------------------

PALETTES = ["viridis", "terrain", "plasma", "magma", "cividis", "turbo",
            "Spectral", "RdYlBu", "coolwarm", "gray"]


def build_palette_luts(n_stops=64):
    """Precompute hex color LUTs for a curated set of colormaps, so the
    browser can switch palettes instantly without a server round trip."""
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    luts = {}
    for name in PALETTES:
        cmap = mpl.colormaps[name]
        luts[name] = [mcolors.to_hex(cmap(t)) for t in np.linspace(0, 1, n_stops)]
    return luts


def compute_hillshade(arr, transform, azimuth=315.0, altitude=45.0):
    """Standard analytical hillshade from a value grid treated as elevation.
    Returns an array in [0, 1] (NaN preserved where input is NaN)."""
    dx = abs(transform.a)
    dy = abs(transform.e)
    if dx == 0:
        dx = 1.0
    if dy == 0:
        dy = 1.0

    gy, gx = np.gradient(arr, dy, dx)

    slope = np.pi / 2.0 - np.arctan(np.sqrt(gx ** 2 + gy ** 2))
    aspect = np.arctan2(-gx, gy)

    az_rad = np.deg2rad(azimuth)
    alt_rad = np.deg2rad(altitude)

    shaded = (
        np.sin(alt_rad) * np.sin(slope)
        + np.cos(alt_rad) * np.cos(slope) * np.cos((az_rad - np.pi / 2.0) - aspect)
    )
    shaded = np.clip(shaded, 0.0, 1.0)
    shaded[~np.isfinite(arr)] = np.nan
    return shaded


def reproject_to_wgs84(arr, shade, transform, crs_wkt, max_dim=260):
    """Reproject the value grid + hillshade to EPSG:4326 so they can be laid
    over a real web map (OSM/satellite tiles expect lat/lon). Returns None
    if the source raster has no CRS — there's nothing to place on a map.

    Returns: dst_arr, dst_shade, (west, south, east, north)
    """
    if crs_wkt is None:
        return None

    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS

    src_crs = CRS.from_wkt(crs_wkt)
    dst_crs = CRS.from_epsg(4326)
    height, width = arr.shape
    west, south, east, north = rasterio.transform.array_bounds(height, width, transform)

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, west, south, east, north
    )
    longer = max(dst_width, dst_height)
    if longer > max_dim:
        factor = max_dim / longer
        dst_width = max(1, int(dst_width * factor))
        dst_height = max(1, int(dst_height * factor))
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, width, height, west, south, east, north,
            dst_width=dst_width, dst_height=dst_height,
        )

    dst_arr = np.full((dst_height, dst_width), np.nan, dtype="float64")
    reproject(
        source=arr, destination=dst_arr,
        src_transform=transform, src_crs=src_crs, src_nodata=np.nan,
        dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    dst_shade = np.full((dst_height, dst_width), np.nan, dtype="float64")
    reproject(
        source=shade, destination=dst_shade,
        src_transform=transform, src_crs=src_crs, src_nodata=np.nan,
        dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    bounds = rasterio.transform.array_bounds(dst_height, dst_width, dst_transform)
    return dst_arr, dst_shade, bounds


# --------------------------------------------------------------------------
# Selection mask
# --------------------------------------------------------------------------

def compute_mask(arr, mode, value=None, threshold=None, min_val=None, max_val=None):
    finite = np.isfinite(arr)
    if mode == "within":
        if threshold is None or threshold < 0:
            raise ValueError("threshold must be >= 0 for 'within' mode")
        return finite & (np.abs(arr - value) <= threshold)
    elif mode == "above":
        return finite & (arr >= value)
    elif mode == "below":
        return finite & (arr <= value)
    elif mode == "range":
        if min_val is None or max_val is None:
            raise ValueError("min and max are required for 'range' mode")
        if min_val > max_val:
            raise ValueError("min must be <= max for 'range' mode")
        return finite & (arr >= min_val) & (arr <= max_val)
    else:
        raise ValueError(f"unknown mode: {mode}")


# --------------------------------------------------------------------------
# Export: vector (GPKG)
# --------------------------------------------------------------------------

def cells_to_geodataframe(arr, mask, transform, crs_wkt, geometry="polygon"):
    import geopandas as gpd
    from shapely.geometry import Polygon, Point

    rows, cols = np.where(mask)
    geoms, records = [], []
    for r, c in zip(rows, cols):
        if geometry == "point":
            x, y = transform * (c + 0.5, r + 0.5)
            geoms.append(Point(x, y))
        else:
            x0, y0 = transform * (c, r)
            x1, y1 = transform * (c + 1, r + 1)
            geoms.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
        records.append({"row": int(r), "col": int(c), "value": float(arr[r, c])})

    return gpd.GeoDataFrame(records, geometry=geoms, crs=crs_wkt)


# --------------------------------------------------------------------------
# Export: raster (GeoTIFF)
# --------------------------------------------------------------------------

def write_binary_tif(arr, mask, transform, crs_wkt, out_path):
    """1 for selected cells, nodata (0) everywhere else."""
    import rasterio
    from rasterio.crs import CRS

    data = np.zeros(arr.shape, dtype="uint8")
    data[mask] = 1
    crs = CRS.from_wkt(crs_wkt) if crs_wkt else None

    profile = dict(
        driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="uint8", crs=crs, transform=transform, nodata=0,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)


def write_masked_tif(arr, mask, transform, crs_wkt, out_path):
    """Original values for selected cells, NaN (nodata) everywhere else."""
    import rasterio
    from rasterio.crs import CRS

    data = np.where(mask, arr, np.nan).astype("float32")
    crs = CRS.from_wkt(crs_wkt) if crs_wkt else None

    profile = dict(
        driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform, nodata=np.nan,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)


# --------------------------------------------------------------------------
# Front end
# --------------------------------------------------------------------------

VIEWER_CSS = r"""
  :root {
    --bg: #14161a; --panel: #1d2026; --text: #e8e9ec;
    --muted: #9aa0ab; --accent: #5ec9ff; --gold: #e8b84b; --accent2: #7bd88f;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; height: 100%; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    overflow: hidden;
  }
  .app { display: flex; flex-direction: column; height: 100vh; }

  .topbar {
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 14px 20px; flex-shrink: 0; gap: 16px;
  }
  .title-group { display: flex; align-items: baseline; gap: 8px; }
  h1 { font-size: 19px; font-weight: 700; margin: 0; white-space: nowrap; }
  .tagline { font-size: 12px; color: var(--muted); margin-top: 2px; }
  .meta { color: var(--muted); font-size: 12px; text-align: right; max-width: 380px; line-height: 1.5; }

  h2 { font-size: 12px; font-weight: 700; margin: 0 0 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }

  .main-area { display: flex; flex-direction: column; flex: 1; min-height: 0; padding: 0 20px 20px; gap: 16px; }
  .top-row { display: flex; flex: 1; min-height: 0; gap: 16px; }

  #viewStack {
    position: relative; flex: 1; min-width: 0; display: flex;
    align-items: center; justify-content: center;
  }
  canvas { background: #0c0d10; border-radius: 10px; display: block; cursor: crosshair; max-width: 100%; max-height: 100%; }
  #mapContainer {
    display: none; border-radius: 10px; background: #0c0d10;
    width: 100%; height: 100%;
  }
  .rover-overlay-img { image-rendering: pixelated; }

  .hover-tooltip {
    position: absolute; display: none; pointer-events: none; z-index: 500;
    background: rgba(15,16,20,0.92); border: 1px solid #3a3d46; color: var(--text);
    font-size: 12px; padding: 5px 9px; border-radius: 6px; white-space: nowrap;
  }

  .panel-toggle-btn {
    flex-shrink: 0; width: 28px; align-self: stretch; background: var(--panel);
    border: none; border-radius: 8px; color: var(--muted); cursor: pointer;
    font-size: 14px; padding: 0;
  }
  .panel-toggle-btn:hover { color: var(--text); background: #262a32; }

  .side-panel {
    width: 300px; flex-shrink: 0; overflow-y: auto; overflow-x: hidden;
    background: var(--panel); border-radius: 10px; padding: 16px;
    transition: width 0.2s ease, opacity 0.15s ease, padding 0.2s ease;
  }
  .side-panel.collapsed {
    width: 0; padding: 16px 0; opacity: 0; pointer-events: none;
  }
  .panel-section { margin-bottom: 18px; }
  .panel-section:last-child { margin-bottom: 0; }

  .bottom-panel {
    flex-shrink: 0; max-height: 260px; overflow-y: auto;
    background: var(--panel); border-radius: 10px; padding: 16px;
    display: flex; gap: 32px; flex-wrap: wrap;
  }
  .bottom-panel-section { flex: 1; min-width: 260px; }
  .bottom-panel-section:last-child { min-width: 320px; }

  .range-inputs { display: flex; align-items: center; gap: 8px; }
  .range-dash { color: var(--muted); flex-shrink: 0; }
  .btn-row { display: flex; gap: 8px; flex-wrap: wrap; }
  .btn-row button { flex: 1; min-width: 160px; margin-bottom: 0; }

  label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  input[type=number], input[type=file], select {
    width: 100%; background: #14161a; border: 1px solid #33363e; color: var(--text);
    border-radius: 6px; padding: 8px 10px; font-size: 14px;
  }
  input[type=number]:focus, select:focus { outline: none; border-color: var(--accent); }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  .inline-check { display: flex; align-items: center; gap: 8px; font-size: 13px; margin-bottom: 8px; }
  .inline-check input { width: auto; }
  .slider-value { font-size: 11px; color: var(--muted); float: right; }
  .row { margin-bottom: 14px; }
  .stat { font-size: 13px; margin-bottom: 6px; }
  .stat b { color: var(--accent); }
  .legend { height: 12px; border-radius: 6px; margin: 6px 0; }
  .legend-labels { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); }
  .hint { font-size: 12px; color: var(--muted); margin-top: 10px; line-height: 1.5; }
  .divider { border-top: 1px solid #2a2d34; margin: 16px 0; }
  button {
    width: 100%; background: var(--accent2); color: #0c1f10; border: none;
    border-radius: 6px; padding: 9px 12px; font-size: 13px; font-weight: 600;
    cursor: pointer; margin-bottom: 8px;
  }
  button.secondary { background: #33363e; color: var(--text); }
  button.gold { background: var(--gold); color: #2a1e04; }
  button:hover { filter: brightness(1.08); }
  button:disabled { opacity: 0.5; cursor: default; }
  .status { font-size: 12px; margin-top: 4px; min-height: 16px; color: var(--muted); }
  .status.error { color: #ff8080; }
  .status.ok { color: var(--accent2); }
  .radio-row { display: flex; gap: 14px; font-size: 13px; margin-bottom: 4px; }
  .radio-row label { display: flex; align-items: center; gap: 6px; color: var(--text); margin-bottom: 0; }
  .radio-row input { width: auto; }
  .btn-group { display: flex; gap: 8px; }
  .btn-group button { margin-bottom: 0; }
"""

VIEWER_JS = r"""
const DATA = __DATA_JSON__;
const rows = DATA.rows, cols = DATA.cols;
const values = DATA.values;
const hillshadeGrid = DATA.hillshade;
const vmin = DATA.vmin, vmax = DATA.vmax;
const palettes = DATA.palettes;
const wgs84 = DATA.wgs84;

let currentPalette = DATA.default_palette;
let opacity = 1.0;
let hillshadeOn = false;
let hillshadeStrength = 0.5;
let basemapOn = false;
let lastCriteria = null;

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function colorForValue(v) {
  if (v === null) return null;
  const lut = palettes[currentPalette];
  let norm = (v - vmin) / (vmax - vmin || 1);
  norm = Math.min(1, Math.max(0, norm));
  const i = Math.round(norm * (lut.length - 1));
  return lut[i];
}

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const cellBase = Math.max(3, Math.min(22, Math.floor(720 / Math.max(rows, cols))));
const gridW = cols * cellBase, gridH = rows * cellBase;
const PAD = Math.ceil(cellBase * 1.6);
canvas.width = gridW + PAD * 2;
canvas.height = gridH + PAD * 2;
canvas.style.width = canvas.width + 'px';
canvas.style.height = canvas.height + 'px';

let currentScale = new Float32Array(rows * cols).fill(1.0);
let hoverTarget = new Float32Array(rows * cols).fill(1.0);
let selectTarget = new Float32Array(rows * cols).fill(1.0);
let selectedMask = new Uint8Array(rows * cols);

const HOVER_SCALE = 1.9, HOVER_MATCH_SCALE = 1.35, SELECT_SCALE = 1.55, EASE = 0.22;

let hoveredR = -1, hoveredC = -1;
let hoverThreshold = parseFloat(__DEFAULT_THRESHOLD__);

function idx(r, c) { return r * cols + c; }
function fmt(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  return Number(v.toFixed(3)).toString();
}

function computeHoverTargets() {
  hoverTarget.fill(1.0);
  if (hoveredR < 0) return 0;
  const hv = values[hoveredR][hoveredC];
  if (hv === null) return 0;
  let matches = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = values[r][c];
      if (v === null) continue;
      if (r === hoveredR && c === hoveredC) { hoverTarget[idx(r, c)] = HOVER_SCALE; matches++; }
      else if (Math.abs(v - hv) <= hoverThreshold) { hoverTarget[idx(r, c)] = HOVER_MATCH_SCALE; matches++; }
    }
  }
  return matches;
}

function matchesCriteria(v, crit) {
  if (v === null) return false;
  if (crit.mode === 'within') return Math.abs(v - crit.value) <= crit.threshold;
  if (crit.mode === 'above') return v >= crit.value;
  if (crit.mode === 'below') return v <= crit.value;
  if (crit.mode === 'range') return v >= crit.min && v <= crit.max;
  return false;
}

function applyCriteria(crit) {
  let count = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = values[r][c];
      const i = idx(r, c);
      const hit = matchesCriteria(v, crit);
      selectedMask[i] = hit ? 1 : 0;
      selectTarget[i] = hit ? SELECT_SCALE : 1.0;
      if (hit) count++;
    }
  }
  return count;
}

function clearSelection() {
  selectedMask.fill(0);
  selectTarget.fill(1.0);
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const order = [];
  for (let i = 0; i < rows * cols; i++) order.push(i);
  order.sort((a, b) => currentScale[a] - currentScale[b]);

  for (const i of order) {
    const r = Math.floor(i / cols), c = i % cols;
    const v = values[r][c];
    const hex = colorForValue(v);
    const s = currentScale[i];
    const isHoveredExact = (r === hoveredR && c === hoveredC);
    const isHoverMatch = hoverTarget[i] > 1.01 && !isHoveredExact;
    const isSelected = selectedMask[i] === 1;

    const x0 = PAD + c * cellBase, y0 = PAD + r * cellBase;
    const cx = x0 + cellBase / 2, cy = y0 + cellBase / 2;
    const size = cellBase * s;

    if (hex === null) {
      ctx.fillStyle = 'rgba(120,120,130,0.35)';
    } else {
      let [rr, gg, bb] = hexToRgb(hex);
      if (hillshadeOn) {
        const sh = hillshadeGrid[r][c];
        if (sh !== null) {
          const shaded = 0.3 + 0.7 * sh;
          const factor = 1 * (1 - hillshadeStrength) + shaded * hillshadeStrength;
          rr *= factor; gg *= factor; bb *= factor;
        }
      }
      ctx.fillStyle = `rgba(${rr | 0}, ${gg | 0}, ${bb | 0}, ${opacity})`;
    }
    if (s > 1.01) {
      ctx.save();
      const glow = isSelected && !isHoveredExact && !isHoverMatch
        ? 'rgba(232, 184, 75, 0.55)' : 'rgba(94, 201, 255, 0.55)';
      ctx.shadowColor = glow;
      ctx.shadowBlur = 10 * (s - 1);
      ctx.fillRect(cx - size / 2, cy - size / 2, size, size);
      ctx.restore();
      ctx.strokeStyle = isSelected && !isHoveredExact && !isHoverMatch
        ? 'rgba(232, 184, 75, 0.9)' : 'rgba(255,255,255,0.85)';
      ctx.lineWidth = 1;
      ctx.strokeRect(cx - size / 2, cy - size / 2, size, size);
    } else {
      ctx.fillRect(cx - size / 2, cy - size / 2, size, size);
    }
  }
}

function animate() {
  for (let i = 0; i < currentScale.length; i++) {
    const target = Math.max(hoverTarget[i], selectTarget[i]);
    const d = target - currentScale[i];
    currentScale[i] = Math.abs(d) > 0.002 ? currentScale[i] + d * EASE : target;
  }
  if (!basemapOn) draw();
  requestAnimationFrame(animate);
}

const viewStack = document.getElementById('viewStack');
const hoverTooltip = document.getElementById('hoverTooltip');

function showTooltip(clientX, clientY, text) {
  const rect = viewStack.getBoundingClientRect();
  hoverTooltip.textContent = text;
  hoverTooltip.style.left = (clientX - rect.left + 14) + 'px';
  hoverTooltip.style.top = (clientY - rect.top + 14) + 'px';
  hoverTooltip.style.display = 'block';
}

function hideTooltip() {
  hoverTooltip.style.display = 'none';
}

canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (canvas.width / rect.width) - PAD;
  const my = (e.clientY - rect.top) * (canvas.height / rect.height) - PAD;
  const c = Math.floor(mx / cellBase), r = Math.floor(my / cellBase);
  if (r < 0 || r >= rows || c < 0 || c >= cols) { hoveredR = -1; hoveredC = -1; }
  else { hoveredR = r; hoveredC = c; }
  const matches = computeHoverTargets();
  const hv = (hoveredR >= 0) ? values[hoveredR][hoveredC] : null;
  document.getElementById('hoveredCell').textContent = hoveredR >= 0 ? `row ${hoveredR}, col ${hoveredC}` : '—';
  document.getElementById('hoveredValue').textContent = fmt(hv);
  document.getElementById('hoverMatchCount').textContent = hoveredR >= 0 && hv !== null ? matches : '—';
  if (hoveredR >= 0 && hv !== null) showTooltip(e.clientX, e.clientY, `Value: ${fmt(hv)}`);
  else hideTooltip();
});

canvas.addEventListener('mouseleave', () => {
  hoveredR = -1; hoveredC = -1;
  computeHoverTargets();
  document.getElementById('hoveredCell').textContent = '—';
  document.getElementById('hoveredValue').textContent = '—';
  document.getElementById('hoverMatchCount').textContent = '—';
  hideTooltip();
});

document.getElementById('hoverThreshold').addEventListener('input', (e) => {
  const parsed = parseFloat(e.target.value);
  hoverThreshold = Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
  const matches = computeHoverTargets();
  if (hoveredR >= 0) document.getElementById('hoverMatchCount').textContent = matches;
});

const legendBar = document.getElementById('legendBar');
function updateLegend() {
  legendBar.style.background = `linear-gradient(to right, ${palettes[currentPalette].join(',')})`;
}
updateLegend();
document.getElementById('legendMin').textContent = fmt(vmin);
document.getElementById('legendMax').textContent = fmt(vmax);

// ---- display controls ----
const paletteSelect = document.getElementById('paletteSelect');
Object.keys(palettes).forEach((name) => {
  const opt = document.createElement('option');
  opt.value = name; opt.textContent = name;
  if (name === currentPalette) opt.selected = true;
  paletteSelect.appendChild(opt);
});
paletteSelect.addEventListener('change', (e) => {
  currentPalette = e.target.value;
  updateLegend();
  if (basemapOn) refreshColorOverlay();
});

const opacitySlider = document.getElementById('opacitySlider');
const opacityValueLabel = document.getElementById('opacityValue');
opacitySlider.addEventListener('input', (e) => {
  opacity = parseInt(e.target.value, 10) / 100;
  opacityValueLabel.textContent = e.target.value + '%';
  if (basemapOn) refreshColorOverlay();
});

const hillshadeToggle = document.getElementById('hillshadeToggle');
const hillshadeStrengthSlider = document.getElementById('hillshadeStrength');
const hillshadeValueLabel = document.getElementById('hillshadeValue');
hillshadeToggle.addEventListener('change', (e) => {
  hillshadeOn = e.target.checked;
  hillshadeStrengthSlider.disabled = !hillshadeOn;
  if (basemapOn) refreshColorOverlay();
});
hillshadeStrengthSlider.addEventListener('input', (e) => {
  hillshadeStrength = parseInt(e.target.value, 10) / 100;
  hillshadeValueLabel.textContent = e.target.value + '%';
  if (basemapOn) refreshColorOverlay();
});

// ---- selection panel ----
const modeSelect = document.getElementById('selMode');
const valueRow = document.getElementById('selValueRow');
const thresholdRow = document.getElementById('selThresholdRow');
const rangeRow = document.getElementById('selRangeRow');
function syncFieldVisibility() {
  const mode = modeSelect.value;
  valueRow.style.display = (mode === 'range') ? 'none' : 'block';
  thresholdRow.style.display = (mode === 'within') ? 'block' : 'none';
  rangeRow.style.display = (mode === 'range') ? 'block' : 'none';
}
modeSelect.addEventListener('change', syncFieldVisibility);
syncFieldVisibility();

function readCriteria() {
  const mode = modeSelect.value;
  if (mode === 'range') {
    const min = parseFloat(document.getElementById('selMin').value);
    const max = parseFloat(document.getElementById('selMax').value);
    if (!Number.isFinite(min) || !Number.isFinite(max)) return { error: 'Enter both a min and a max.' };
    if (min > max) return { error: 'Min must be less than or equal to max.' };
    return { mode, min, max };
  }
  const value = parseFloat(document.getElementById('selValue').value);
  const threshold = parseFloat(document.getElementById('selThreshold').value);
  if (!Number.isFinite(value)) return { error: 'Enter a numeric value.' };
  if (mode === 'within' && (!Number.isFinite(threshold) || threshold < 0)) {
    return { error: 'Enter a threshold of 0 or greater for "within".' };
  }
  return { mode, value, threshold: mode === 'within' ? threshold : null };
}

const selStatus = document.getElementById('selStatus');

document.getElementById('selectBtn').addEventListener('click', () => {
  const crit = readCriteria();
  selStatus.className = 'status';
  if (crit.error) { selStatus.textContent = crit.error; selStatus.className = 'status error'; return; }
  lastCriteria = crit;
  const count = applyCriteria(crit);
  selStatus.textContent = `${count} cell${count === 1 ? '' : 's'} selected`;
  selStatus.className = 'status ok';
  if (basemapOn) refreshSelectionOverlay();
});

document.getElementById('clearSelectionBtn').addEventListener('click', () => {
  lastCriteria = null;
  clearSelection();
  selStatus.textContent = 'Selection cleared';
  selStatus.className = 'status';
  if (basemapOn) refreshSelectionOverlay();
});

// ---- basemap (map view) ----
let leafletMap = null;
let colorOverlayLayer = null;
let selectionOverlayLayer = null;

function cellColorRGBA(v, shadeVal) {
  const hex = colorForValue(v);
  if (hex === null) return null;
  let [rr, gg, bb] = hexToRgb(hex);
  if (hillshadeOn && shadeVal !== null && shadeVal !== undefined) {
    const shaded = 0.3 + 0.7 * shadeVal;
    const factor = (1 - hillshadeStrength) + shaded * hillshadeStrength;
    rr *= factor; gg *= factor; bb *= factor;
  }
  return [rr | 0, gg | 0, bb | 0];
}

function buildColorOverlayDataUrl() {
  const w = wgs84.cols, h = wgs84.rows;
  const off = document.createElement('canvas');
  off.width = w; off.height = h;
  const octx = off.getContext('2d');
  const imgData = octx.createImageData(w, h);
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const i = r * w + c;
      const v = wgs84.values[r][c];
      const rgba = cellColorRGBA(v, wgs84.hillshade[r][c]);
      const p = i * 4;
      if (rgba === null) {
        imgData.data[p + 3] = 0;
      } else {
        imgData.data[p] = rgba[0];
        imgData.data[p + 1] = rgba[1];
        imgData.data[p + 2] = rgba[2];
        imgData.data[p + 3] = Math.round(opacity * 255);
      }
    }
  }
  octx.putImageData(imgData, 0, 0);
  return off.toDataURL();
}

function buildSelectionOverlayDataUrl(crit) {
  const w = wgs84.cols, h = wgs84.rows;
  const off = document.createElement('canvas');
  off.width = w; off.height = h;
  const octx = off.getContext('2d');
  const imgData = octx.createImageData(w, h);
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const v = wgs84.values[r][c];
      const hit = matchesCriteria(v, crit);
      const p = (r * w + c) * 4;
      if (hit) {
        imgData.data[p] = 232; imgData.data[p + 1] = 184; imgData.data[p + 2] = 75; imgData.data[p + 3] = 190;
      } else {
        imgData.data[p + 3] = 0;
      }
    }
  }
  octx.putImageData(imgData, 0, 0);
  return off.toDataURL();
}

function wgs84Bounds() {
  const b = wgs84.bounds;
  return [[b.south, b.west], [b.north, b.east]];
}

function wgs84RowColFromLatLng(lat, lng) {
  const b = wgs84.bounds;
  if (lng < b.west || lng > b.east || lat < b.south || lat > b.north) return null;
  const col = Math.floor((lng - b.west) / (b.east - b.west) * wgs84.cols);
  const row = Math.floor((b.north - lat) / (b.north - b.south) * wgs84.rows);
  return {
    row: Math.min(wgs84.rows - 1, Math.max(0, row)),
    col: Math.min(wgs84.cols - 1, Math.max(0, col)),
  };
}

function onMapMouseMove(e) {
  const rc = wgs84RowColFromLatLng(e.latlng.lat, e.latlng.lng);
  if (!rc) { onMapMouseOut(); return; }
  const v = wgs84.values[rc.row][rc.col];
  const clientX = e.originalEvent ? e.originalEvent.clientX : null;
  const clientY = e.originalEvent ? e.originalEvent.clientY : null;
  document.getElementById('hoveredCell').textContent = `row ${rc.row}, col ${rc.col} (map)`;
  document.getElementById('hoveredValue').textContent = fmt(v);
  if (v === null) {
    document.getElementById('hoverMatchCount').textContent = '—';
    hideTooltip();
    return;
  }
  if (clientX !== null) showTooltip(clientX, clientY, `Value: ${fmt(v)}`);
  let matches = 0;
  for (let r = 0; r < wgs84.rows; r++) {
    for (let c = 0; c < wgs84.cols; c++) {
      const vv = wgs84.values[r][c];
      if (vv !== null && Math.abs(vv - v) <= hoverThreshold) matches++;
    }
  }
  document.getElementById('hoverMatchCount').textContent = matches;
}

function onMapMouseOut() {
  document.getElementById('hoveredCell').textContent = '—';
  document.getElementById('hoveredValue').textContent = '—';
  document.getElementById('hoverMatchCount').textContent = '—';
  hideTooltip();
}

function initMap() {
  if (leafletMap || !wgs84) return;
  leafletMap = L.map('mapContainer', { preferCanvas: true });
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(leafletMap);

  const bounds = wgs84Bounds();
  leafletMap.fitBounds(bounds);
  colorOverlayLayer = L.imageOverlay(buildColorOverlayDataUrl(), bounds, {
    opacity: 1, className: 'rover-overlay-img',
  }).addTo(leafletMap);

  leafletMap.on('mousemove', onMapMouseMove);
  leafletMap.on('mouseout', onMapMouseOut);
}

function refreshColorOverlay() {
  if (!colorOverlayLayer) return;
  colorOverlayLayer.setUrl(buildColorOverlayDataUrl());
}

function refreshSelectionOverlay() {
  if (!leafletMap) return;
  if (!lastCriteria) {
    if (selectionOverlayLayer) { leafletMap.removeLayer(selectionOverlayLayer); selectionOverlayLayer = null; }
    return;
  }
  const url = buildSelectionOverlayDataUrl(lastCriteria);
  const bounds = wgs84Bounds();
  if (selectionOverlayLayer) selectionOverlayLayer.setUrl(url);
  else selectionOverlayLayer = L.imageOverlay(url, bounds, { opacity: 1, className: 'rover-overlay-img' }).addTo(leafletMap);
}

const basemapToggle = document.getElementById('basemapToggle');
const basemapHint = document.getElementById('basemapHint');
const mapContainerEl = document.getElementById('mapContainer');

if (!wgs84) {
  basemapToggle.disabled = true;
  basemapHint.textContent = 'No CRS found in this raster — basemap unavailable.';
} else {
  basemapHint.textContent = '';
}

basemapToggle.addEventListener('change', (e) => {
  basemapOn = e.target.checked;
  if (basemapOn) {
    canvas.style.display = 'none';
    mapContainerEl.style.display = 'block';
    initMap();
    setTimeout(() => {
      leafletMap.invalidateSize();
      refreshColorOverlay();
      refreshSelectionOverlay();
    }, 50);
  } else {
    canvas.style.display = 'block';
    mapContainerEl.style.display = 'none';
  }
});

// ---- export ----
const exportStatus = document.getElementById('exportStatus');

async function doExport(kind, btn) {
  const crit = readCriteria();
  exportStatus.className = 'status';
  if (crit.error) { exportStatus.textContent = crit.error; exportStatus.className = 'status error'; return; }

  const geometry = document.querySelector('input[name=geomType]:checked').value;
  btn.disabled = true;
  exportStatus.textContent = 'Exporting…';

  try {
    const resp = await fetch('/export', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        mode: crit.mode, value: crit.value, threshold: crit.threshold,
        min: crit.min, max: crit.max, kind, geometry,
      })
    });
    if (!resp.ok) {
      let msg = resp.statusText;
      try { const err = await resp.json(); msg = err.error || msg; } catch (e) {}
      exportStatus.textContent = 'Error: ' + msg;
      exportStatus.className = 'status error';
      return;
    }
    const blob = await resp.blob();
    const cd = resp.headers.get('Content-Disposition') || '';
    const m = cd.match(/filename="?([^"]+)"?/);
    const filename = m ? m[1] : 'export.dat';
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    exportStatus.textContent = `Downloaded ${filename}`;
    exportStatus.className = 'status ok';
  } catch (e) {
    exportStatus.textContent = 'Error: ' + e.message;
    exportStatus.className = 'status error';
  } finally {
    btn.disabled = false;
  }
}

document.getElementById('exportVectorBtn').addEventListener('click', (e) => doExport('vector', e.target));
document.getElementById('exportBinaryBtn').addEventListener('click', (e) => doExport('binary_tif', e.target));
document.getElementById('exportMaskedBtn').addEventListener('click', (e) => doExport('masked_tif', e.target));

// ---- upload ----
const uploadStatus = document.getElementById('uploadStatus');
document.getElementById('uploadBtn').addEventListener('click', async () => {
  const fileInput = document.getElementById('tifFile');
  const file = fileInput.files[0];
  uploadStatus.className = 'status';
  if (!file) { uploadStatus.textContent = 'Choose a .tif file first.'; uploadStatus.className = 'status error'; return; }

  const formData = new FormData();
  formData.append('file', file);
  uploadStatus.textContent = 'Uploading…';
  try {
    const resp = await fetch('/upload', { method: 'POST', body: formData });
    const result = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      uploadStatus.textContent = 'Error: ' + (result.error || resp.statusText);
      uploadStatus.className = 'status error';
      return;
    }
    uploadStatus.textContent = 'Loaded — refreshing…';
    uploadStatus.className = 'status ok';
    setTimeout(() => window.location.reload(), 400);
  } catch (e) {
    uploadStatus.textContent = 'Error: ' + e.message;
    uploadStatus.className = 'status error';
  }
});

// ---- side panel toggle ----
const sidePanel = document.getElementById('sidePanel');
const panelToggle = document.getElementById('panelToggle');
panelToggle.addEventListener('click', () => {
  sidePanel.classList.toggle('collapsed');
  panelToggle.textContent = sidePanel.classList.contains('collapsed') ? '▸' : '◂';
  setTimeout(() => { if (leafletMap) leafletMap.invalidateSize(); }, 220);
});

draw();
animate();
"""

PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Raster Rover Exploration Viewer</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>__CSS__</style>
</head>
<body>

<div class="app">
  <div class="topbar">
    <div class="title-group">
      <h1>🌍 Raster Rover Exploration Viewer</h1>
      <div class="tagline">Explore by hover. Extract by selection.</div>
    </div>
    <div class="meta">__META__</div>
  </div>

  <div class="main-area">
    <div class="top-row">
      <div id="viewStack">
        <canvas id="canvas"></canvas>
        <div id="mapContainer"></div>
        <div class="hover-tooltip" id="hoverTooltip"></div>
      </div>

      <button id="panelToggle" class="panel-toggle-btn" title="Toggle panel">◂</button>

      <div class="side-panel" id="sidePanel">

        <div class="panel-section">
          <h2>Display</h2>
          <div class="row">
            <div class="inline-check">
              <input type="checkbox" id="basemapToggle">
              <label for="basemapToggle" style="margin:0;" id="basemapLabel">Show basemap</label>
            </div>
            <div class="hint" id="basemapHint" style="margin-top:0;"></div>
          </div>
          <div class="row">
            <label for="paletteSelect">Color palette</label>
            <select id="paletteSelect"></select>
          </div>
          <div class="row">
            <label for="opacitySlider">Opacity <span class="slider-value" id="opacityValue">100%</span></label>
            <input type="range" id="opacitySlider" min="0" max="100" value="100">
          </div>
          <div class="row">
            <div class="inline-check">
              <input type="checkbox" id="hillshadeToggle">
              <label for="hillshadeToggle" style="margin:0;">Hillshade</label>
            </div>
            <label for="hillshadeStrength">Strength <span class="slider-value" id="hillshadeValue">50%</span></label>
            <input type="range" id="hillshadeStrength" min="0" max="100" value="50" disabled>
          </div>
        </div>

        <div class="divider"></div>

        <div class="panel-section">
          <h2>Hover to explore</h2>
          <div class="row">
            <label for="hoverThreshold">Hover match threshold (value units)</label>
            <input type="number" id="hoverThreshold" step="__STEP__" value="__DEFAULT_THRESHOLD__">
          </div>
          <div class="row">
            <div class="stat">Hovered cell: <b id="hoveredCell">—</b></div>
            <div class="stat">Value: <b id="hoveredValue">—</b></div>
            <div class="stat">Matches: <b id="hoverMatchCount">—</b></div>
          </div>
          <div class="row">
            <label>Value range</label>
            <div class="legend" id="legendBar"></div>
            <div class="legend-labels"><span id="legendMin"></span><span id="legendMax"></span></div>
          </div>
        </div>

        <div class="divider"></div>

        <div class="panel-section">
          <h2>Select to extract</h2>
          <div class="row">
            <label for="selMode">Mode</label>
            <select id="selMode">
              <option value="within">Cells within threshold of value</option>
              <option value="above">Cells above value</option>
              <option value="below">Cells below value</option>
              <option value="range">Cells within a range (min–max)</option>
            </select>
          </div>
          <div class="row" id="selValueRow">
            <label for="selValue">Value</label>
            <input type="number" id="selValue" step="any" placeholder="e.g. 950">
          </div>
          <div class="row" id="selThresholdRow">
            <label for="selThreshold">Threshold</label>
            <input type="number" id="selThreshold" step="any" placeholder="e.g. 25">
          </div>
          <div class="row" id="selRangeRow" style="display:none;">
            <label>Range</label>
            <div class="range-inputs">
              <input type="number" id="selMin" step="any" placeholder="Min">
              <span class="range-dash">–</span>
              <input type="number" id="selMax" step="any" placeholder="Max">
            </div>
          </div>
          <div class="btn-group row">
            <button id="selectBtn" class="gold">Select</button>
            <button id="clearSelectionBtn" class="secondary">Clear</button>
          </div>
          <div class="status" id="selStatus"></div>
        </div>

      </div>
    </div>

    <div class="bottom-panel" id="bottomPanel">
      <div class="bottom-panel-section">
        <h2>Load a raster</h2>
        <div class="row">
          <label for="tifFile">Add your own .tif</label>
          <input type="file" id="tifFile" accept=".tif,.tiff">
        </div>
        <button id="uploadBtn" class="secondary">Add .tif</button>
        <div class="status" id="uploadStatus"></div>
      </div>

      <div class="bottom-panel-section">
        <h2>Export selection</h2>
        <div class="row">
          <label>Vector geometry</label>
          <div class="radio-row">
            <label><input type="radio" name="geomType" value="polygon" checked> Polygons</label>
            <label><input type="radio" name="geomType" value="point"> Points</label>
          </div>
        </div>
        <div class="btn-row">
          <button id="exportVectorBtn">Export vector (.gpkg)</button>
          <button id="exportBinaryBtn" class="secondary">Export binary raster (.tif) — 1 / NA</button>
          <button id="exportMaskedBtn" class="secondary">Export masked raster (.tif) — values / NA</button>
        </div>
        <div class="status" id="exportStatus"></div>
        <div class="hint">
          Exports use the Mode/Value/Threshold (or Range) from Select to extract, whether or not you've clicked Select.
        </div>
      </div>
    </div>
  </div>
</div>

<script>
__VIEWER_JS__
</script>
</body>
</html>
"""


def _wgs84_payload(arr, shade, transform, crs_wkt):
    if crs_wkt is None:
        return None
    try:
        result = reproject_to_wgs84(arr, shade, transform, crs_wkt)
    except Exception as e:
        print(f"Warning: couldn't reproject raster for basemap view: {e}", file=sys.stderr)
        return None
    if result is None:
        return None
    dst_arr, dst_shade, (west, south, east, north) = result

    values = [[None if not np.isfinite(v) else round(float(v), 6) for v in row] for row in dst_arr]
    hillshade = [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in dst_shade]

    return {
        "rows": dst_arr.shape[0], "cols": dst_arr.shape[1],
        "values": values, "hillshade": hillshade,
        "bounds": {"west": west, "south": south, "east": east, "north": north},
    }


def _viewer_data(arr, cmap_name, transform, crs_wkt):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
        if vmin == vmax:
            vmax = vmin + 1.0

    values = [[None if not np.isfinite(v) else round(float(v), 6) for v in row] for row in arr]

    shade = compute_hillshade(arr, transform)
    hillshade = [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in shade]

    palettes = build_palette_luts()
    default_palette = cmap_name if cmap_name in palettes else "viridis"

    wgs84 = _wgs84_payload(arr, shade, transform, crs_wkt)

    return {
        "rows": arr.shape[0], "cols": arr.shape[1],
        "values": values,
        "hillshade": hillshade,
        "vmin": round(vmin, 6), "vmax": round(vmax, 6),
        "palettes": palettes,
        "default_palette": default_palette,
        "wgs84": wgs84,
    }


def build_page_html(arr, crs_display, source_name, cmap_name, transform, crs_wkt):
    data = _viewer_data(arr, cmap_name, transform, crs_wkt)
    vmin, vmax = data["vmin"], data["vmax"]
    value_range = (vmax - vmin) if vmax > vmin else 1.0
    default_threshold = round(value_range * 0.05, 6)
    step = round(max(value_range / 200.0, 1e-6), 6)

    meta = f"{source_name} &middot; {arr.shape[0]}&times;{arr.shape[1]} cells &middot; CRS: {crs_display}"

    html = PAGE_TEMPLATE
    html = html.replace("__CSS__", VIEWER_CSS)
    html = html.replace("__META__", meta)
    html = html.replace("__STEP__", str(step))
    html = html.replace("__DEFAULT_THRESHOLD__", str(default_threshold))

    js = VIEWER_JS.replace("__DATA_JSON__", json.dumps(data))
    js = js.replace("__DEFAULT_THRESHOLD__", str(default_threshold))
    html = html.replace("__VIEWER_JS__", js)
    return html


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------

def create_app(initial_arr, initial_transform, initial_crs_wkt, initial_crs_display,
               initial_source_name, cmap_name, max_dim, band):
    from flask import Flask, Response, request, jsonify, send_file
    import os

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB upload cap

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "file too large (100MB limit)"}), 413

    state = {
        "arr": initial_arr, "transform": initial_transform,
        "crs_wkt": initial_crs_wkt, "crs_display": initial_crs_display,
        "source_name": initial_source_name, "cmap_name": cmap_name,
        "max_dim": max_dim, "band": band,
    }

    @app.route("/")
    def index():
        html = build_page_html(state["arr"], state["crs_display"], state["source_name"], state["cmap_name"], state["transform"], state["crs_wkt"])
        return Response(html, mimetype="text/html")

    @app.route("/upload", methods=["POST"])
    def upload():
        f = request.files.get("file")
        if f is None or f.filename == "":
            return jsonify({"error": "no file received"}), 400
        if not f.filename.lower().endswith((".tif", ".tiff")):
            return jsonify({"error": "please upload a .tif or .tiff file"}), 400

        tmp_dir = tempfile.mkdtemp(prefix="raster_rover_upload_")
        tmp_path = os.path.join(tmp_dir, f.filename)
        f.save(tmp_path)

        try:
            arr, crs_display, transform, crs_wkt = read_raster(
                tmp_path, band=state["band"], max_dim=state["max_dim"]
            )
        except Exception as e:
            return jsonify({"error": f"couldn't read that file as a raster: {e}"}), 400
        finally:
            try:
                os.remove(tmp_path)
                os.rmdir(tmp_dir)
            except OSError:
                pass

        state["arr"] = arr
        state["transform"] = transform
        state["crs_wkt"] = crs_wkt
        state["crs_display"] = crs_display
        state["source_name"] = f.filename

        return jsonify({"ok": True})

    @app.route("/export", methods=["POST"])
    def export():
        payload = request.get_json(force=True, silent=True) or {}
        mode = payload.get("mode")
        value = payload.get("value")
        threshold = payload.get("threshold")
        min_val = payload.get("min")
        max_val = payload.get("max")
        kind = payload.get("kind")
        geometry = payload.get("geometry", "polygon")

        if mode not in ("within", "above", "below", "range"):
            return jsonify({"error": "mode must be 'within', 'above', 'below', or 'range'"}), 400

        if mode == "range":
            if min_val is None or max_val is None:
                return jsonify({"error": "min and max are required for 'range' mode"}), 400
            try:
                min_val = float(min_val)
                max_val = float(max_val)
            except (TypeError, ValueError):
                return jsonify({"error": "min and max must be numbers"}), 400
            if min_val > max_val:
                return jsonify({"error": "min must be <= max"}), 400
        else:
            if value is None:
                return jsonify({"error": "value is required"}), 400
            try:
                value = float(value)
            except (TypeError, ValueError):
                return jsonify({"error": "value must be a number"}), 400
            if mode == "within":
                if threshold is None:
                    return jsonify({"error": "threshold is required for 'within' mode"}), 400
                try:
                    threshold = float(threshold)
                except (TypeError, ValueError):
                    return jsonify({"error": "threshold must be a number"}), 400
                if threshold < 0:
                    return jsonify({"error": "threshold must be >= 0"}), 400

        arr = state["arr"]
        transform = state["transform"]
        crs_wkt = state["crs_wkt"]

        try:
            mask = compute_mask(arr, mode, value=value, threshold=threshold, min_val=min_val, max_val=max_val)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        count = int(mask.sum())
        if count == 0:
            return jsonify({"error": "no cells matched that criteria"}), 404

        tmp_dir = tempfile.mkdtemp(prefix="raster_rover_export_")
        if mode == "range":
            safe_v = str(min_val).replace(".", "p").replace("-", "neg")
            safe_t = str(max_val).replace(".", "p").replace("-", "neg")
        else:
            safe_v = str(value).replace(".", "p").replace("-", "neg")
            safe_t = str(threshold).replace(".", "p") if threshold is not None else "na"

        if kind == "vector":
            if geometry not in ("polygon", "point"):
                return jsonify({"error": "geometry must be 'polygon' or 'point'"}), 400
            out_name = f"rover_{mode}_v{safe_v}_t{safe_t}_{geometry}.gpkg"
            out_path = os.path.join(tmp_dir, out_name)
            gdf = cells_to_geodataframe(arr, mask, transform, crs_wkt, geometry=geometry)
            gdf.to_file(out_path, driver="GPKG", layer="matched_cells")
            mimetype = "application/geopackage+sqlite3"
        elif kind == "binary_tif":
            out_name = f"rover_{mode}_v{safe_v}_t{safe_t}_binary.tif"
            out_path = os.path.join(tmp_dir, out_name)
            write_binary_tif(arr, mask, transform, crs_wkt, out_path)
            mimetype = "image/tiff"
        elif kind == "masked_tif":
            out_name = f"rover_{mode}_v{safe_v}_t{safe_t}_masked.tif"
            out_path = os.path.join(tmp_dir, out_name)
            write_masked_tif(arr, mask, transform, crs_wkt, out_path)
            mimetype = "image/tiff"
        else:
            return jsonify({"error": "kind must be 'vector', 'binary_tif', or 'masked_tif'"}), 400

        response = send_file(out_path, mimetype=mimetype, as_attachment=True, download_name=out_name)

        @response.call_on_close
        def _cleanup():
            try:
                os.remove(out_path)
                os.rmdir(tmp_dir)
            except OSError:
                pass

        return response

    return app


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Raster Rover Exploration Viewer")
    parser.add_argument("raster", nargs="?", help="Path to a GeoTIFF raster (defaults to a built-in demo DEM if omitted)")
    parser.add_argument("--band", type=int, default=1, help="Band number to read (default 1)")
    parser.add_argument("--max-dim", type=int, default=220, help="Max grid dimension in cells (downsamples larger rasters)")
    parser.add_argument("--cmap", default=None, help="Matplotlib colormap name (default: terrain for demo, viridis otherwise)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default 8765)")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open a browser tab")
    args = parser.parse_args()

    if args.raster:
        try:
            arr, crs_display, transform, crs_wkt = read_raster(args.raster, band=args.band, max_dim=args.max_dim)
        except Exception as e:
            print(f"Couldn't read '{args.raster}' as a raster: {e}", file=sys.stderr)
            sys.exit(1)
        source_name = Path(args.raster).name
        cmap_name = args.cmap or "viridis"
    else:
        arr, crs_display, transform, crs_wkt = make_demo_dem()
        source_name = "Demo DEM (built-in example — add your own .tif above)"
        cmap_name = args.cmap or "terrain"
        print("No raster given — starting with the built-in demo DEM. "
              "Use the 'Add .tif' button in the browser to load your own.")

    app = create_app(arr, transform, crs_wkt, crs_display, source_name, cmap_name, args.max_dim, args.band)
    url = f"http://{args.host}:{args.port}/"

    if not args.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print(f"Raster Rover serving at {url}  (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
