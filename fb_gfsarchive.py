"""fb_gfsarchive — any GFS 0.25 deg cycle since 2021, a few records at a time.

WHY THIS EXISTS. fb_gfsbox reads NOMADS' grib filter, and NOMADS keeps about
ten days. A verification that means anything has to reach across seasons, and
a case study has to reach the day the case happened, so this module reads the
same files from NOAA's open-data archive instead:

    https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.YYYYMMDD/HH/atmos/
        gfs.tHHz.pgrb2.0p25.fFFF         (and its .idx)

The .idx sidecar lists every record's byte offset, so a handful of fields is a
handful of HTTP range requests -- about 1 MB a record -- never the 500 MB file.

SAME ANSWER SHAPE AS fb_gfsbox.fetch: {'lon', 'lat', 'cycle', **fields}, lat
ascending, lon ascending in signed degrees. The global records are cached once
per (cycle, fhr) and cropped per call, so four scoring windows cost one
download. Unlike fb_gfsbox, an extent may cross the antimeridian: give it as
lon0 < lon1 with lon1 past 180 (e.g. 150..230) and the grid is rolled to suit.

No key, no registration; the bucket is public (NOAA Open Data Dissemination).
"""
import logging
import os
import tempfile
from datetime import timezone

import numpy as np
import requests

import fb_gfsbox as gb

log = logging.getLogger("fb_gfsarchive")

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
_UA = {"User-Agent": "FrostByte/1.0"}


def url_for(cycle, fhr):
    return (f"{BUCKET}/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/"
            f"gfs.t{cycle:%H}z.pgrb2.0p25.f{int(fhr):03d}")


def _cache_dir():
    base = os.environ.get("FROSTBYTE_CACHE") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cache")
    d = os.path.join(base, "gfsarchive")
    os.makedirs(d, exist_ok=True)
    return d


def _ranges(idx_text, fields):
    """{field name: (start, end)} byte ranges out of one .idx listing."""
    lines = [ln.split(":") for ln in idx_text.splitlines() if ln.strip()]
    out = {}
    for f in fields:
        want = f.level.replace("_", " ")
        for i, parts in enumerate(lines):
            if len(parts) > 4 and parts[3] == f.var and parts[4] == want:
                start = int(parts[1])
                end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else ""
                out[f.name] = (start, end)
                break
    return out


def _global(cycle, fhr, fields, timeout=120):
    """The global records for `fields`, from cache or the bucket. The cache
    file GROWS: a later caller asking for more fields adds to it."""
    path = os.path.join(_cache_dir(), f"gfs_{cycle:%Y%m%d%H}_f{int(fhr):03d}.npz")
    have = {}
    if os.path.exists(path):
        try:
            with np.load(path) as d:
                have = {k: d[k] for k in d.files}
        except Exception as e:                                 # noqa: BLE001
            log.info("GFS archive: unreadable cache (%s); refetching.", e)
            have = {}
    need = [f for f in fields if f.name not in have]
    if not need:
        return have
    url = url_for(cycle, fhr)
    r = requests.get(url + ".idx", headers=_UA, timeout=timeout)
    if r.status_code != 200:
        log.warning("GFS archive: no index for %s f%03d (HTTP %s).",
                    cycle.strftime("%Y%m%d%H"), fhr, r.status_code)
        return None
    rng = _ranges(r.text, need)
    missing = [f.name for f in need if f.name not in rng]
    if missing:
        log.warning("GFS archive: %s f%03d has no %s.",
                    cycle.strftime("%Y%m%d%H"), fhr, ", ".join(missing))
        return None
    tf = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    try:
        for f in need:
            a, b = rng[f.name]
            rr = requests.get(url, headers={**_UA, "Range": f"bytes={a}-{b}"},
                              timeout=timeout)
            if rr.status_code not in (200, 206):
                log.warning("GFS archive: range read failed (HTTP %s).",
                            rr.status_code)
                return None
            tf.write(rr.content)
        tf.close()
        lon, lat, got = gb._decode(tf.name, need)
    finally:
        try:
            tf.close()
            os.unlink(tf.name)
        except OSError:
            pass
    have.update(got)
    have["lon"], have["lat"] = lon, lat
    np.savez_compressed(path, **have)
    log.info("GFS archive: %s f%03d +%d record(s).",
             cycle.strftime("%Y%m%d%H"), fhr, len(need))
    return have


def fetch(extent, fields, cache_tag=None, fhr=0, cycle=None, pad=0.0, **_kw):
    """fb_gfsbox.fetch's answer from the archive. `cycle` is required (a
    timezone-aware or naive-UTC datetime); `cache_tag` is accepted and ignored,
    because this cache is keyed on the cycle alone and cropped per call."""
    if cycle is None:
        raise ValueError("fb_gfsarchive.fetch needs a pinned cycle")
    if cycle.tzinfo is None:
        cycle = cycle.replace(tzinfo=timezone.utc)
    g = _global(cycle, fhr, fields)
    if g is None:
        return None
    lon0, lon1, lat0, lat1 = (float(v) for v in extent)
    lon0, lon1, lat0, lat1 = lon0 - pad, lon1 + pad, lat0 - pad, lat1 + pad
    glon = np.asarray(g["lon"], dtype="float64")
    lat = np.asarray(g["lat"], dtype="float64")
    # Roll the 0..360 grid so it starts at lon0, then keep lon0..lon1.
    rolled = np.mod(glon - lon0, 360.0) + lon0
    order = np.argsort(rolled, kind="stable")
    rolled = rolled[order]
    jx = order[rolled <= lon1 + 1e-9]
    iy = np.where((lat >= lat0 - 1e-9) & (lat <= lat1 + 1e-9))[0]
    out = {f.name: np.asarray(g[f.name])[np.ix_(iy, jx)] for f in fields}
    out.update(lon=rolled[rolled <= lon1 + 1e-9], lat=lat[iy], cycle=cycle)
    return out
