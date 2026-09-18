#!/usr/bin/env python3
"""fb_gfsbox.py — one GFS analysis box from NOMADS, for the basin charts.

WHAT IT IS. tropical_shear_atl.fetch_gfs_shear does exactly the right thing
for one product: ask the NOMADS grib-filter for a lat/lon subset of the newest
GFS f000, cache it per cycle, walk back through older cycles when the newest
is not posted, and back off quietly when NOMADS is rate-limiting rather than
mistaking a holding page for a missing run. All of that is worth keeping and
none of it is about shear. This is the same machinery with the FIELDS made an
argument, so the dry-air chart and the steering chart do not each grow their
own copy of the retry logic and their own subtly different idea of what a 302
from NOMADS means.

WHAT IT RETURNS. {'lon', 'lat', <your names>, 'cycle'} — plain numpy, latitude
ascending, one 2-D array per requested field. The caller names the fields it
wants; nothing here knows what a relative humidity is for.

THE RATE LIMIT, because it is the failure that actually happens: these charts
run in a background rota that has usually just finished a NOMADS-heavy job, and
a throttled NOMADS answers 302 to an HTML holding page, or 403, not 404. Read
as "this cycle isn't posted" that walks the fetch back through five cycles in
as many seconds and gives up on a GFS that is perfectly healthy. So a refusal
is told apart from an absence: one quiet 25-second backoff, then older cycles
are served from cache only.
"""
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

log = logging.getLogger("fb_gfsbox")

NOMADS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
GEFS_CHEM_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_chem_0p25.pl"
PAD_DEG = 4.0            # past the map window, so a fill meets the frame
CACHE_SWEEP_S = 24 * 3600

# The grib-filter level token for a column-integrated field, spelled the way
# the CGI wants it before quoting.
WHOLE_ATMOS = "entire_atmosphere_(considered_as_a_single_layer)"


class Source:
    """One NOMADS grib-filter endpoint: which CGI, which directory and file
    name a cycle maps to, and how long after cycle time the run lands.

    The lag is the part worth getting right. Asking for a run that is not
    posted yet is not an error anyone sees — the fetch just walks back a cycle
    — but starting three cycles too early means three wasted round trips
    against a service that rate-limits, every single run.
    """

    def __init__(self, filter_url, dir_tmpl, file_tmpl, cycle_hours=6,
                 lag_h=3.5, name="GFS"):
        self.filter_url = filter_url
        self.dir_tmpl = dir_tmpl
        self.file_tmpl = file_tmpl
        self.cycle_hours = int(cycle_hours)
        self.lag_h = float(lag_h)
        self.name = name

    def newest_cycle(self, now=None):
        """The newest cycle whose analysis hour should be on NOMADS."""
        now = now or datetime.now(timezone.utc)
        base = (now - timedelta(hours=self.lag_h)).replace(
            minute=0, second=0, microsecond=0)
        return base.replace(hour=(base.hour // self.cycle_hours)
                            * self.cycle_hours)

    def url_for(self, cyc, fhr):
        return (self.dir_tmpl.format(cyc=cyc),
                self.file_tmpl.format(cyc=cyc, fhr=fhr))


# GFS 0.25 deg, the suite's workhorse analysis: f000 lands about 3.5 h after
# cycle time (the same number tropical_shear_atl walks back from).
GFS = Source(NOMADS_FILTER,
             "%2Fgfs.{cyc:%Y%m%d}%2F{cyc:%H}%2Fatmos",
             "gfs.t{cyc:%H}z.pgrb2.0p25.f{fhr:03d}")

# GEFS-Aerosols (the chem member), 0.25 deg, four cycles a day like GFS but
# posted later — the aerosol run trails the atmospheric one, and at 03Z the
# day's 00Z chem directory is still a 500 from NOMADS while the GFS 00Z run
# has been up for hours. Seven hours is measured, not assumed.
GEFS_CHEM = Source(GEFS_CHEM_FILTER,
                   "%2Fgefs.{cyc:%Y%m%d}%2F{cyc:%H}%2Fchem%2Fpgrb2ap25",
                   "gefs.chem.t{cyc:%H}z.a2d_0p25.f{fhr:03d}.grib2",
                   cycle_hours=6, lag_h=7.0, name="GEFS-Aerosols")


class Field:
    """One field to pull: the filter's var/level tokens, the GRIB short name
    it decodes to, and the name the caller wants it back under.

    `keys` is the escape hatch for records eccodes has no parameter name for.
    GEFS-Aerosols is all of them: every aerosol record decodes as shortName
    'unknown', and the only thing telling dust from sea salt from sulfate is
    the GRIB2 constituentType, with the wavelength picking the band. cfgrib
    cannot address those, so a field carrying `keys` is read message by
    message with eccodes instead.
    """

    def __init__(self, name, var, level, short, isobaric=None, keys=None):
        self.name = name
        self.var = var
        self.level = level
        self.short = short
        self.isobaric = isobaric
        self.keys = dict(keys or {})

    def __repr__(self):                                        # pragma: no cover
        return f"<Field {self.name} {self.var}@{self.level}>"


def isobaric(name, var, short, mb):
    """A field on a pressure level, e.g. isobaric('rh700', 'RH', 'r', 700)."""
    return Field(name, var, f"{mb}_mb", short, isobaric=mb)


def column(name, var, short):
    """A whole-atmosphere field, e.g. column('pwat', 'PWAT', 'pwat')."""
    return Field(name, var, WHOLE_ATMOS, short)


# WMO code table 4.233 aerosol types, as GEFS-Aerosols writes them. Only the
# ones this suite has a use for; dust is the whole point of the SAL chart, and
# the total is what a satellite radiometer would see.
AEROSOL_TOTAL = 62000
AEROSOL_DUST = 62001


def aerosol(name, constituent=AEROSOL_DUST, wavelength=545):
    """One aerosol optical-depth band from GEFS-Aerosols, by constituent.

    `wavelength` is the GRIB scaled value (545 = the 550 nm band everybody
    quotes AOD at; the file also carries 338, 430, 620, 841, 1628 and 11000).
    """
    return Field(name, "AOTK", "entire_atmosphere", None,
                 keys={"constituentType": int(constituent),
                       "scaledValueOfFirstWavelength": int(wavelength),
                       "parameterNumber": 102})


def _newest_cycle(now=None, source=None):
    """The newest cycle of `source` (GFS by default) that should be posted."""
    return (source or GFS).newest_cycle(now)


def _query(cyc, fields, extent, fhr=0, source=None):
    lon0, lon1, lat0, lat1 = extent
    levels = sorted({f.level for f in fields})
    variables = sorted({f.var for f in fields})
    lev_q = "".join("&lev_" + lv.replace("(", "%28").replace(")", "%29")
                    .replace(" ", "_") + "=on" for lv in levels)
    var_q = "".join(f"&var_{v}=on" for v in variables)
    d, fn = (source or GFS).url_for(cyc, fhr)
    return (f"?dir={d}&file={fn}{var_q}{lev_q}"
            f"&subregion=&leftlon={lon0:.1f}&rightlon={lon1:.1f}"
            f"&toplat={lat1:.1f}&bottomlat={lat0:.1f}")


def _axes_from_keys(ec, gid):
    """(lon, lat) for a regular lat/lon message, built from the grid
    definition — NOT from eccodes' distinctLongitudes.

    This is the bug that would have shifted every dust plume: a NOMADS subset
    of a western-hemisphere box comes back as 280 deg -> 360 deg, and the last
    column is written as 0. distinctLongitudes SORTS that, so it hands back
    [0, 280, 280.25, ...] while the values are still in scan order — the map
    would be drawn with one column of the African coast pasted at the far west
    edge and everything else off by a cell. Built from first-point plus
    increment, the axis matches the data; converted to signed degrees and
    argsorted, it also matches the convention the basin windows are written
    in.
    """
    ni = ec.codes_get(gid, "Ni")
    nj = ec.codes_get(gid, "Nj")
    lon0 = float(ec.codes_get(gid, "longitudeOfFirstGridPointInDegrees"))
    lat0 = float(ec.codes_get(gid, "latitudeOfFirstGridPointInDegrees"))
    di = float(ec.codes_get(gid, "iDirectionIncrementInDegrees"))
    dj = float(ec.codes_get(gid, "jDirectionIncrementInDegrees"))
    if ec.codes_get(gid, "iScansNegatively"):
        di = -di
    if not ec.codes_get(gid, "jScansPositively"):
        dj = -dj
    lon = (lon0 + di * np.arange(ni, dtype="float64")) % 360.0
    lat = lat0 + dj * np.arange(nj, dtype="float64")
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return lon, lat


def _decode_keyed(path, fields):
    """Read fields addressed by raw GRIB keys, message by message, with
    eccodes. Returns (lon, lat, {name: 2-D array}) for those it found."""
    import eccodes as ec
    out, lat, lon = {}, None, None
    fh = open(path, "rb")
    try:
        while True:
            gid = ec.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                for f in fields:
                    if f.name in out:
                        continue
                    ok = True
                    for k, want in f.keys.items():
                        try:
                            ok = ok and ec.codes_get(gid, k) == want
                        except Exception:                      # noqa: BLE001
                            ok = False
                        if not ok:
                            break
                    if not ok:
                        continue
                    lon, lat = _axes_from_keys(ec, gid)
                    out[f.name] = np.asarray(
                        ec.codes_get_values(gid), dtype="float32"
                    ).reshape(lat.size, lon.size)
            finally:
                ec.codes_release(gid)
    finally:
        fh.close()
    if lon is not None and lon.size > 1 and np.any(np.diff(lon) < 0):
        order = np.argsort(lon)
        lon = lon[order]
        out = {k: v[:, order] for k, v in out.items()}
    return lon, lat, out


def _decode(path, fields):
    """Pull every requested field out of one grib2 file. Uses cfgrib's
    open_datasets (not a single filter_by_keys) because a request that mixes
    pressure levels with a whole-atmosphere field is two hypercubes, and one
    open_dataset call can only ever see one of them. Fields carrying raw GRIB
    keys take the eccodes path instead (see Field)."""
    out = {}
    lat = lon = None
    keyed = [f for f in fields if f.keys]
    fields = [f for f in fields if not f.keys]
    if keyed:
        lon, lat, got = _decode_keyed(path, keyed)
        out.update(got)
    if not fields:
        missing = [f.name for f in keyed if f.name not in out]
        if missing or lat is None:
            raise KeyError(f"GRIB file has no {', '.join(missing) or 'grid'}")
        if lat.size > 1 and lat[0] > lat[-1]:
            lat = lat[::-1]
            out = {k: v[::-1] for k, v in out.items()}
        return lon, lat, out
    # Imported here and not at the top of the function: a request that is all
    # aerosol records never needs cfgrib at all, and loading it costs the
    # eccodes bindings for nothing.
    import cfgrib
    dss = cfgrib.open_datasets(path, backend_kwargs=dict(indexpath=""))
    try:
        for f in fields:
            for ds in dss:
                if f.short not in ds:
                    continue
                da = ds[f.short]
                if f.isobaric is not None:
                    if "isobaricInhPa" not in da.dims and \
                            float(da.coords.get("isobaricInhPa", -1)) != f.isobaric:
                        continue
                    if "isobaricInhPa" in da.dims:
                        da = da.sel(isobaricInhPa=f.isobaric)
                out[f.name] = np.asarray(da.values, dtype="float32")
                lat = np.asarray(ds["latitude"].values, dtype="float64")
                lon = np.asarray(ds["longitude"].values, dtype="float64")
                break
    finally:
        for ds in dss:
            try:
                ds.close()
            except Exception:                                  # noqa: BLE001
                pass
    missing = [f.name for f in fields + keyed if f.name not in out]
    if missing or lat is None:
        raise KeyError(f"GFS file has no {', '.join(missing) or 'grid'}")
    if lat.size > 1 and lat[0] > lat[-1]:          # GRIB runs N->S; flip
        lat = lat[::-1]
        out = {k: v[::-1] for k, v in out.items()}
    return lon, lat, out


def _trim_antimeridian(lon, fields):
    """Drop any column sitting exactly on +-180 degrees.

    Cartopy projects a filled contour as a polygon, and a polygon whose edge
    lies exactly on the PlateCarree boundary comes back from shapely as a
    GeometryCollection, which cartopy then tries to subscript:

        TypeError: 'GeometryCollection' object is not subscriptable

    It is data-dependent -- the same chart renders from one GFS cycle and dies
    on the next, depending on whether a contour happens to close on the edge --
    which makes it exactly the kind of failure that would have shown up as a
    blank E. Pacific panel some mornings and not others. The GFS grid puts a
    point ON 180 for the two windows that reach the dateline; OISST's does not
    (its cells are centered on .125), which is why only the model charts ever
    hit this. A quarter degree of fill at the frame edge is the price.
    """
    lon = np.asarray(lon, dtype="float64")
    edge = np.isclose(np.abs(lon), 180.0)
    if not edge.any():
        return lon, fields
    keep = ~edge
    trimmed = {k: (v[..., keep] if np.ndim(v) >= 2
                   and np.shape(v)[-1] == lon.size else v)
               for k, v in fields.items()}
    return lon[keep], trimmed


def _cache_path(cache_tag, cyc, fhr):
    return os.path.join(tempfile.gettempdir(),
                        f"fb_gfsbox_{cache_tag}_{cyc:%Y%m%d%H}_f{fhr:03d}.npz")


def _sweep(cache_tag):
    tmp = tempfile.gettempdir()
    now = time.time()
    try:
        for fn in os.listdir(tmp):
            if fn.startswith(f"fb_gfsbox_{cache_tag}_") and fn.endswith(".npz"):
                p = os.path.join(tmp, fn)
                if now - os.path.getmtime(p) > CACHE_SWEEP_S:
                    os.remove(p)
    except OSError:
        pass


def fetch(extent, fields, cache_tag, fhr=0, timeout=90, cycles_back=5,
          pad=PAD_DEG, source=None, cycle=None):
    """{'lon', 'lat', 'cycle', **fields} from the newest usable GFS run, or
    None.

    `cache_tag` must be unique per (basin window, field set): the cache keys on
    it and the cycle, NOT on the extent, so two callers sharing a tag would
    hand each other the wrong ocean — the same rule fetch_gfs_shear keeps.

    `cycle` pins the run instead of starting from the newest one, and is what a
    VERIFICATION needs: scoring a forecast means holding an old cycle's f012
    against a later cycle's f000, and a fetcher that can only reach the newest
    run can only ever compare a model with itself. The walk-back still applies
    from there, so a pinned cycle NOMADS has already rotated off falls through
    to the one before it rather than failing.
    """
    lon0, lon1, lat0, lat1 = (float(v) for v in extent)
    box = (max(-180.0, lon0 - pad), min(180.0, lon1 + pad),
           max(-85.0, lat0 - pad), min(85.0, lat1 + pad))
    source = source or GFS
    names = [f.name for f in fields]
    newest = cycle or source.newest_cycle()
    throttle_retry_left, net_ok, k = 1, True, 0

    while k < cycles_back:
        cyc = newest - timedelta(hours=source.cycle_hours * k)
        path = _cache_path(cache_tag, cyc, fhr)
        try:
            if os.path.exists(path):
                d = np.load(path)
                if all(n in d for n in names):
                    res = {n: d[n].copy() for n in names}
                    lon, res = _trim_antimeridian(d["lon"].copy(), res)
                    res.update(lon=lon, lat=d["lat"].copy(), cycle=cyc)
                    d.close()
                    log.info("GFS box: cache hit (%s).", os.path.basename(path))
                    return res
                d.close()
        except Exception as e:                                 # noqa: BLE001
            log.info("GFS box: unreadable cache (%s); refetching.", e)
        if not net_ok:
            k += 1
            continue

        url = source.filter_url + _query(cyc, fields, box, fhr, source)
        log.info("%s box: trying %s f%03d via the grib filter...",
                 source.name, cyc.strftime("%Y%m%d%H"), fhr)
        try:
            # allow_redirects=False on purpose: a rate-limited NOMADS answers
            # 302 to an HTML page, which followed silently reads as a short
            # body and gets logged as "cycle not posted".
            r = requests.get(url, headers={"User-Agent": "FrostByte/1.0"},
                             timeout=timeout, allow_redirects=False)
        except Exception as e:                                 # noqa: BLE001
            log.warning("GFS box: %s request failed (%s).", cyc, e)
            k += 1
            continue

        refused = (300 <= r.status_code < 400 or r.status_code in (403, 429)
                   or r.status_code >= 500
                   or (r.status_code == 200 and r.content[:1] == b"<"))
        if refused:
            if throttle_retry_left:
                throttle_retry_left -= 1
                log.info("GFS box: NOMADS is throttling us (HTTP %s), not a "
                         "missing cycle; going quiet 25s, then retrying.",
                         r.status_code)
                time.sleep(25)
                continue
            log.info("GFS box: still throttled (HTTP %s); older cycles from "
                     "cache only.", r.status_code)
            net_ok = False
            k += 1
            continue
        if r.status_code != 200 or len(r.content) < 1500:
            log.info("GFS box: %s f%03d not available (HTTP %s).",
                     cyc.strftime("%Y%m%d%H"), fhr, r.status_code)
            k += 1
            continue

        tf = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
        tf.write(r.content)
        tf.close()
        try:
            lon, lat, got = _decode(tf.name, fields)
        except Exception as e:                                 # noqa: BLE001
            log.warning("Couldn't read the %s GFS file (%s); trying an "
                        "earlier run.", cyc.strftime("%Y%m%d%H"), e)
            k += 1
            continue
        finally:
            try:
                os.unlink(tf.name)
            except OSError:
                pass

        try:
            np.savez_compressed(path, lon=lon, lat=lat, **got)
            _sweep(cache_tag)
        except OSError:
            pass
        log.info("GFS box fetched (cycle %s, grid %s).",
                 cyc.strftime("%Y%m%d%H"), next(iter(got.values())).shape)
        lon, res = _trim_antimeridian(lon, dict(got))
        res.update(lon=lon, lat=lat, cycle=cyc)
        return res

    log.error("None of the last %d %s runs had the fields this chart needs; "
              "try again in a little while.", cycles_back, source.name)
    return None
