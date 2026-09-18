"""Boxology run forward — the trajectory the technique already implies.

WHAT THIS ADDS TO fb_boxology. That module answers "how hard is the layer being
warmed HERE, NOW", and shades it. This one carries the answer along the flow:
the isobars are the geostrophic streamlines, so a parcel arriving anywhere can
be traced back up its own isobar channel and the thickness contours it crossed
on the way counted. That count -- ``boxes`` below -- is the accumulated
advection in the technique's own currency, and a forecaster can check any
single number in it with a pencil.

WHY THE TRAJECTORY IS THE SAME IDENTITY. fb_boxology's derivation is an exact
statement about the thickness field's material derivative,

    dh/dt  =  -V . grad(h)  =  -V0 . grad(h)

with V0 the 1000 hPa geostrophic wind, because the thermal wind blows ALONG the
thickness contours and drops out. Integrating that along a path instead of
evaluating it at a point introduces no new physics and no new data -- only the
frozen-pattern assumption, said out loud at the bottom.

READ THIS BEFORE YOU BUILD A FORECAST OUT OF IT. The obvious next move is to
call -V.grad(h) a temperature forecast: multiply the rate by a time step, draw
where the 540 line went, ship it. It does not work, and boxology_verify.py
measures how badly, on real GFS grids:

  * The advection itself is measured FAITHFULLY. Against the model's own
    full-wind layer-mean advection the boxology correlates at +0.93 -- from two
    contour families, with no wind field at all. The identity survives contact
    with a real atmosphere.

  * And the identity's own corollary shows up in the data. Every level's
    geostrophic wind differs from V0 only by a multiple of the thermal wind,
    and the thermal wind cannot advect thickness, so the geostrophic thickness
    advection is the SAME at every level. The boxology should therefore match
    ANY level's advection, and match it best wherever the real wind is closest
    to geostrophic. It does: slope 1.04 against 500 mb, 0.75 against 850 mb,
    and only 0.32 against 1000 mb -- the level friction ruins. Its V0 (7.4 m/s
    rms) is nearly the model's 850 mb wind (7.2), not its 1000 mb wind (5.2).

  * But advection is NOT the local tendency. Against a centered 6-hour
    thickness tendency the regression slope is 0.31: advection over-predicts
    the actual change by more than three times, steady from 40 to 330 km
    smoothing. About two thirds of every advective signal is canceled by
    adiabatic cooling from the ascent that the warm advection itself forces.
    This is the atmosphere, not the technique -- the model's own total
    advection is canceled just as hard (slope 0.33).

  * So the kinematic forecast has NEGATIVE skill against persistence: SS -0.74
    at 6 h, -0.87 at 12 h, -2.3 at 24 h. Persistence wins at every lead.

  * The one cheerful part, and it is a real one: the boxology's GEOSTROPHIC
    advection tracks the net tendency BETTER than the model's total advection
    does (+0.50 against +0.41). Restricting the identity to the geostrophic
    wind is not a compromise. The ageostrophic advection it omits is
    disproportionately the part that gets adiabatically canceled, so leaving it
    out moves the answer toward the truth.

WHAT THAT LEAVES, and it is not nothing. ``advance`` is a validated measure of
TRANSPORT and of ACCUMULATED ADVECTION: where this air came from, and how many
thickness intervals of warming the flow has done to it on the way. It is not a
forecast of the thickness at the arrival point, because the parcel's thickness
is not conserved -- two thirds of what the flow does to it gets undone. The
functions here that return a thickness (``thk``, ``line_forecast``) exist to
make that measurement and to let the next person reproduce the negative result;
none of them is wired into a product. A coefficient regressed against GFS to
rescue them would be exactly the corrected guess fb_boxology refuses when it
fades rather than applying a gradient-wind ratio.

THE PREMISE TRAVELS WITH THE PARCEL. fb_boxology fades its shading wherever the
geostrophic premise fails -- near the equator, in sharply curved flow, over high
ground -- and refuses to put a number in a fade ("no number in fades", Evan,
2026-09-16). A trajectory makes that rule strictly harder to keep, because a
parcel arriving at a perfectly good point may have spent six hours crossing the
Rockies to get there. So the taper is sampled at every substep and the MINIMUM
along the path is what gates the answer: an answer is only as trustworthy as the
worst ground its air came over. Points whose trajectory passed through a fade,
left the domain, or entered the equatorial belt carry nothing and say which --
``status``, read with STATUS_TEXT.

THE ASSUMPTION THAT IS NOT IN THE IDENTITY, said out loud: the pressure pattern
is held still for the length of the run. Real troughs move, and past about 12
hours that -- not the advection -- is the leading error in the transport too.
"""
import logging

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

import fb_boxology as bx

# Kelvin of layer-mean temperature per dam of 1000-500 hPa thickness: the
# hypsometric relation read the other way round, dT = (g / (R ln2)) dh.
# 0.493 K/dam, so the house 6 dam interval is 2.96 K -- fb_boxology's dt_line.
K_PER_DAM = bx.G / (bx.R_D * bx.LN2) * 10.0

# THE BOLUS: the quantum's name (Evan, 2026-09-18).
#
# A BOX is the geometry -- the quadrilateral two isobars and two thickness
# lines cut out of the chart. A BOLUS is the fixed amount of airmass change
# that box carries, and the two are worth separate words because keeping them
# apart IS the technique's central claim: the box's AREA varies from 40 km to
# 500 km across, and the bolus inside it does not vary at all.
#
# Borrowed from medicine, where a bolus is a discrete dose delivered all at
# once by a flow -- which is exactly what this is. The wind administers them.
# One bolus is one isobar interval by one thickness interval; on the house
# 6 dam ladder that is 2.96 K of layer-mean temperature.
BOLUS_NAME = 'bolus'

# Trajectory resolution. A semi-Lagrangian back-trajectory has no CFL limit to
# respect -- it is stable at any step -- so the substep count buys PATH
# accuracy and nothing else, and the thing it has to resolve is the curve of
# the isobar the parcel is running down. One grid cell per substep tracks a
# 0.25-deg channel closely enough that halving it moves the 540 line by less
# than the line's own width; the ceiling stops a 250 kt jet on a fine mesh from
# asking for thousands.
STEP_CELLS = 1.0
MIN_SUBSTEPS, MAX_SUBSTEPS = 2, 240

# Poleward clamp for the trajectory: 1/cos(lat) runs away at the pole exactly
# as 1/f does at the equator, and a parcel that walks over the pole has its
# longitude become meaningless. Trajectories are held below this and the
# arrival flagged, rather than allowed to spiral.
LAT_CLAMP = 87.0

# The canonical thickness lines, in dam. 540 is the rain/snow line of first
# resort, 528 the one that means it, 570 the warm-sector marker -- the same
# trio fb_meso_global draws bold, so a forecast line lands on the family the
# chart already wears.
CANON_LINES = (528.0, 540.0, 570.0)

(STATUS_OK, STATUS_FADED, STATUS_OFFGRID, STATUS_BELT,
 STATUS_POLE) = range(5)
STATUS_TEXT = {STATUS_OK: 'traced',
               STATUS_FADED: 'withheld (air came over a fade)',
               STATUS_OFFGRID: 'withheld (air came from off the chart)',
               STATUS_BELT: 'withheld (trajectory entered the tropics)',
               STATUS_POLE: 'withheld (trajectory reached the pole)'}


def _axes(lon2d, lat2d):
    """The 1-D lon/lat axes behind a regular mesh, plus the wrap flag.

    The trajectory has to sample fields at points that are not grid nodes, and
    the cheap way to do that on this suite's meshes is fractional indices into
    a rectilinear grid. So the mesh is CHECKED rather than assumed: a rotated
    or curvilinear mesh raises instead of being silently interpolated as though
    its rows were parallels, which would bend every trajectory by the mesh's
    own skew and still return a plausible-looking chart.

    Returns (lon1d, lat1d, wrap) with wrap True when the mesh closes on itself
    in longitude, so a global trajectory can cross the seam instead of
    stopping at it.
    """
    lon = np.asarray(lon2d, dtype='float64')
    lat = np.asarray(lat2d, dtype='float64')
    if lon.ndim != 2 or lon.shape != lat.shape:
        raise ValueError("boxology forecast: lon2d/lat2d must be matching 2-D meshes")
    if lon.shape[0] < 3 or lon.shape[1] < 3:
        raise ValueError("boxology forecast: mesh too small to trace a trajectory")
    # Unwrapped, so a mesh that crosses the dateline gives a monotone axis
    # instead of a 360-degree cliff in one column.
    lon1d = np.unwrap(np.radians(lon[0, :])) * 180.0 / np.pi
    lat1d = lat[:, 0]
    if not np.allclose(np.diff(lon, axis=0), 0.0, atol=1e-6):
        raise ValueError("boxology forecast: mesh is curvilinear (longitude varies by row)")
    if not np.allclose(np.diff(lat, axis=1), 0.0, atol=1e-6):
        raise ValueError("boxology forecast: mesh is curvilinear (latitude varies by column)")
    # Rectilinear to a tenth of a cell: tighter than any resampling artifact in
    # the suite's own grids, loose enough for float noise in a built mesh.
    dlon = np.diff(lon1d)
    dlat = np.diff(lat1d)
    if np.ptp(dlon) > 0.1 * np.abs(np.mean(dlon)):
        raise ValueError("boxology forecast: longitude axis is not regular")
    if np.ptp(dlat) > 0.1 * np.abs(np.mean(dlat)):
        raise ValueError("boxology forecast: latitude axis is not regular")
    step = float(np.mean(dlon))
    if step <= 0:
        raise ValueError("boxology forecast: longitude axis must ascend")
    # WRAP means the mesh closes on itself, so a trajectory may cross the seam.
    # The test is ">= a full circle", not "== one", because the globe's mesh is
    # seam-PADDED (-180.5..180.25) and covers 360.75 degrees: it closes, with
    # four duplicated columns to spare. An equality test called that mesh
    # regional and stopped every global trajectory dead at the dateline.
    wrap = bool((lon1d[-1] - lon1d[0]) + step >= 360.0 - 0.51 * step)
    return lon1d, lat1d, wrap


def geostrophic_flow(lon2d, lat2d, mslp_hpa, mslp_sigma=3.0):
    """The 1000 hPa geostrophic wind (u, v) in m/s, off the isobars alone.

    V0 = (g/f) k x grad(Z0), with Z0 the 1000 hPa height the chart's isobars
    stand for -- fb_boxology.HPA_TO_GPM back through the same reduction the
    pipeline drew them from, so this wind blows along the isobars actually on
    the chart and not along a differently smoothed copy of them.

    Masked inside fb_boxology's equatorial belt, where 1/f is not a wind.
    """
    lat = np.asarray(lat2d, dtype='float64')
    p = np.asarray(mslp_hpa, dtype='float64')
    if mslp_sigma:
        p = gaussian_filter(p, sigma=mslp_sigma)
    z0 = (p - 1000.0) * bx.HPA_TO_GPM
    dx, dy = bx._metric(lon2d, lat)
    dzdx = np.gradient(z0, axis=1) / dx
    dzdy = np.gradient(z0, axis=0) / dy
    belt = np.abs(lat) < bx.LAT_FADE_LO
    f = 2.0 * bx.OMEGA * np.sin(np.radians(lat))
    f = np.where(belt, np.nan, f)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = -(bx.G / f) * dzdy
        v = (bx.G / f) * dzdx
    return (np.ma.masked_invalid(np.ma.masked_where(belt, u)),
            np.ma.masked_invalid(np.ma.masked_where(belt, v)))


def tendency(lon2d, lat2d, mslp_hpa, thk_dam, **kw):
    """Thickness tendency dh/dt in dam/hour -- fb_boxology's own field, in the
    units the forecast is drawn in.

    This is advection_field() pushed back through the hypsometric relation, and
    it exists to be checked against advance(): a short step of the trajectory
    operator has to reproduce this times the step, or the two halves of the
    module disagree about the same physics. Positive is thickening (warming).
    """
    adv = bx.advection_field(lon2d, lat2d, mslp_hpa, thk_dam, **kw)   # K/day
    return adv / K_PER_DAM / 24.0


def _seam_pad(field, wrap):
    """One duplicated column on the right, so bilinear sampling can span the
    dateline exactly instead of clamping to the last column.

    Without it a global trajectory arriving between the last node and the first
    reads the last node twice -- a quarter-degree flat spot down the whole
    dateline, which is small everywhere and visible as a kink in a 540 line
    that happens to cross it.
    """
    f = np.asarray(field, dtype='float64')
    return np.concatenate([f, f[:, :1]], axis=1) if wrap else f


def _sample(field, row, col, fill=np.nan):
    """Bilinear sample of a seam-padded grid at fractional (row, col).

    'nearest' rather than 'constant': every out-of-range point has already
    been flagged by the caller, so what the clamp returns is discarded, and a
    NaN read here would poison the trajectory's own state on the way to being
    thrown away.
    """
    return map_coordinates(field, np.stack([row, col]), order=1,
                           mode='nearest', cval=fill, prefilter=False)


def advance(lon2d, lat2d, mslp_hpa, thk_dam, hours, *,
            substeps=None, dthk_dam=bx.DEF_DTHK_DAM,
            mslp_sigma=3.0, thk_sigma=0.0,
            curvature_taper=True, elev_m=None,
            seeds=None, keep_path=False, along=None):
    """Run the boxology forward ``hours`` and report the thickness it carries.

    Semi-Lagrangian: every arrival point is traced BACK along the frozen
    geostrophic flow and handed the thickness it finds at its departure point.
    Backwards, not forwards, because backwards lands an answer on every node of
    the chart -- a forward march scatters parcels and leaves holes exactly
    where the flow is diffluent, which on a thickness chart is the warm sector
    somebody is trying to forecast.

    Returns a dict of fields on the input mesh:

        thk      float64  forecast thickness, dam (NaN where no forecast)
        dthk     float64  forecast change, dam
        dtemp    float64  the same change as layer-mean temperature, K
        boxes    float64  the change in THICKNESS INTERVALS -- contour
                          crossings along the trajectory, signed warm positive.
                          This is the hand-auditable number.
        boluses  float64  the same number under the quantum's own name; see
                          BOLUS_NAME for why the two words are both kept.
        status   int8     STATUS_OK, or why this node carries no forecast
        taper    float64  the minimum premise taper met along the trajectory
        back_lon float64  where each arrival point's air came from
        back_lat float64

    ``hours`` may be negative, which runs the operator backwards and is how
    the verification builds a hindcast from a single analysis.

    ``seeds`` is an optional (lon, lat) pair of arrays to trace INSTEAD of
    every grid node, any shape; the returned fields then carry that shape. It
    costs what the seeds cost rather than what the mesh costs, which is what
    makes a chart of a few hundred labelled parcels affordable on a grid where
    tracing all 125,000 of them would not be.

    ``keep_path`` additionally returns ``path_lon`` / ``path_lat``, shaped
    (substeps + 1, ...) — the whole trajectory rather than only where it
    ended, which is what it takes to DRAW one. Off by default because on a
    full mesh it is one array per substep: a 0.25-degree hemisphere at 61
    substeps is about 120 MB of path nobody asked for.

    ``along`` is a field on the mesh to AVERAGE ALONG each trajectory, and it
    is what a question about the air's history needs. It returns
    ``along_mean`` (the time-mean of that field over the path) and
    ``along_frac`` (what fraction of the substeps actually contributed, so a
    caller can tell a full answer from a trajectory that left the grid
    halfway). Handing the land/sea mask to it gives the fraction of its
    recent life each parcel has spent over water — the Bergeron maritime /
    continental prefix, which is a statement about where air came FROM and
    cannot be read off any local field. It accumulates a running sum rather
    than storing the path, so it costs one array instead of `substeps` of
    them and works on a full mesh where ``keep_path`` would not.
    """
    lon1d, lat1d, wrap = _axes(lon2d, lat2d)
    lat = np.asarray(lat2d, dtype='float64')
    thk = np.asarray(thk_dam, dtype='float64')
    if thk_sigma:
        thk = gaussian_filter(thk, sigma=thk_sigma)

    u, v = geostrophic_flow(lon2d, lat2d, mslp_hpa, mslp_sigma)
    # The premise tapers, from the diagnostic's own core so the forecast fades
    # on exactly the ground the shading fades on.
    _adv, taper, belt, _parts = bx._advection_parts(
        lon2d, lat2d, mslp_hpa, thk_dam, mslp_sigma, thk_sigma,
        curvature_taper, elev_m)

    # A masked wind is a wind of zero for the trajectory's purposes -- the
    # parcel stops rather than jumping -- and the belt/taper flags are what
    # actually withhold the answer there.
    uf = np.nan_to_num(np.ma.filled(u, 0.0))
    vf = np.nan_to_num(np.ma.filled(v, 0.0))
    tf = np.nan_to_num(np.asarray(taper, dtype='float64'))
    beltf = np.asarray(belt, dtype='float64')
    u_s, v_s = _seam_pad(uf, wrap), _seam_pad(vf, wrap)
    t_s, b_s = _seam_pad(tf, wrap), _seam_pad(beltf, wrap)
    thk_s = _seam_pad(thk, wrap)
    a_s = None if along is None else _seam_pad(
        np.nan_to_num(np.asarray(along, dtype='float64')), wrap)

    dlon = float(np.mean(np.diff(lon1d)))
    dlat = float(np.mean(np.diff(lat1d)))
    nrow, ncol_s = thk_s.shape
    if substeps is None:
        # Path accuracy only (see STEP_CELLS): size the step so a fast parcel
        # moves about one cell. The 99th percentile rather than the max keeps
        # one bad gradient at a masked edge from setting the cost of the whole
        # field.
        cell_m = abs(dlat) * bx.EARTH_R * np.pi / 180.0
        speed = np.sqrt(uf * uf + vf * vf)
        fast = float(np.percentile(speed[np.isfinite(speed)], 99)) if speed.size else 0.0
        need = abs(hours) * 3600.0 * max(fast, 1.0) / (STEP_CELLS * max(cell_m, 1.0))
        substeps = int(np.clip(np.ceil(need), MIN_SUBSTEPS, MAX_SUBSTEPS))
    substeps = max(int(substeps), 1)
    dt = -float(hours) * 3600.0 / substeps        # seconds, negative = upstream

    def to_index(glon, glat):
        """Fractional (row, col) into the seam-padded arrays.

        Longitude goes through a mod 360, which is what makes one formula serve
        a global mesh, a dateline-crossing regional mesh and a mesh the parcel
        has walked off the west edge of. On a closed mesh the result is always
        in range, so a global trajectory is never flagged off-grid for crossing
        the seam; on a regional one a parcel that leaves lands far out of range
        and is flagged, which is what should happen to it.
        """
        return ((glat - lat1d[0]) / dlat,
                np.mod(glon - lon1d[0], 360.0) / dlon)

    # Trajectory state in degrees, and the flags it collects on the way. The
    # taper starts at the ARRIVAL point's own value, not at 1: a parcel sitting
    # still over the Colorado Plateau has not earned an answer just because it
    # never went anywhere.
    if seeds is None:
        plon = np.array(np.asarray(lon2d, dtype='float64'), copy=True)
        plat = np.array(lat, copy=True)
        tmin = np.array(tf, copy=True)
        hit_belt = beltf > 0.5
        hit_pole = np.abs(lat) >= LAT_CLAMP
        arrive_thk = thk
        off = np.zeros(thk.shape, dtype=bool)
    else:
        # Seeded: every one of those starting values has to be SAMPLED at the
        # seed instead of read off a node, or a parcel released over the
        # Rockies would start life with the taper of whatever grid point the
        # mesh happens to begin at.
        plon = np.array(np.asarray(seeds[0], dtype='float64'), copy=True)
        plat = np.array(np.asarray(seeds[1], dtype='float64'), copy=True)
        _r, _c = to_index(plon, plat)
        _rc = np.clip(_r, 0.0, nrow - 1.0)
        _cc = np.clip(_c, 0.0, ncol_s - 1.0)
        tmin = _sample(t_s, _rc, _cc)
        hit_belt = _sample(b_s, _rc, _cc) > 0.5
        hit_pole = np.abs(plat) >= LAT_CLAMP
        arrive_thk = _sample(thk_s, _rc, _cc)
        off = ~((_r >= 0.0) & (_r <= nrow - 1.0)
                & (_c >= 0.0) & (_c <= ncol_s - 1.0))
    path = [(plon.copy(), plat.copy())] if keep_path else None
    if a_s is None:
        a_sum = a_n = None
    else:
        _r0, _c0 = to_index(plon, plat)
        a_sum = _sample(a_s, np.clip(_r0, 0.0, nrow - 1.0),
                        np.clip(_c0, 0.0, ncol_s - 1.0))
        a_n = np.ones(plon.shape)

    deg = 180.0 / np.pi
    for _ in range(substeps):
        live = ~(off | hit_belt | hit_pole)
        if not live.any():
            break
        r, c = to_index(plon, plat)
        rc = np.clip(r, 0.0, nrow - 1.0)
        u1 = _sample(u_s, rc, np.clip(c, 0.0, ncol_s - 1.0))
        v1 = _sample(v_s, rc, np.clip(c, 0.0, ncol_s - 1.0))
        # RK2 midpoint: the parcel is running down a curving isobar, and Euler
        # cuts every corner to the inside of the curve -- which around a trough
        # walks the trajectory steadily across isobars it should be following,
        # and turns a 12-hour forecast of the 540 line into a forecast of a
        # slightly different 540 line somewhere downstream.
        cosl = np.cos(np.radians(np.clip(plat, -LAT_CLAMP, LAT_CLAMP)))
        mlon = plon + 0.5 * dt * u1 / (bx.EARTH_R * cosl) * deg
        mlat = plat + 0.5 * dt * v1 / bx.EARTH_R * deg
        rm, cm = to_index(mlon, mlat)
        rm = np.clip(rm, 0.0, nrow - 1.0)
        cm = np.clip(cm, 0.0, ncol_s - 1.0)
        u2 = _sample(u_s, rm, cm)
        v2 = _sample(v_s, rm, cm)
        cosm = np.cos(np.radians(np.clip(mlat, -LAT_CLAMP, LAT_CLAMP)))
        nlon = plon + dt * u2 / (bx.EARTH_R * cosm) * deg
        nlat = plat + dt * v2 / bx.EARTH_R * deg

        plon = np.where(live, nlon, plon)
        plat = np.where(live, nlat, plat)
        if path is not None:
            path.append((plon.copy(), plat.copy()))

        # Where the parcel now is, and what that costs it.
        r, c = to_index(plon, plat)
        inside = (r >= 0.0) & (r <= nrow - 1.0) & (c >= 0.0) & (c <= ncol_s - 1.0)
        off |= live & ~inside
        hit_pole |= live & (np.abs(plat) >= LAT_CLAMP)
        ok = live & inside
        rc = np.clip(r, 0.0, nrow - 1.0)
        cc = np.clip(c, 0.0, ncol_s - 1.0)
        tmin = np.where(ok, np.minimum(tmin, _sample(t_s, rc, cc)), tmin)
        hit_belt |= ok & (_sample(b_s, rc, cc) > 0.5)
        if a_s is not None:
            # Only while the parcel is still being traced: a trajectory that
            # left the grid must not go on averaging the edge cell it stopped
            # against, or every starved parcel comes back reading whatever
            # surface the frame happens to end on.
            a_sum = np.where(ok, a_sum + _sample(a_s, rc, cc), a_sum)
            a_n = np.where(ok, a_n + 1.0, a_n)

    r, c = to_index(plon, plat)
    departed = _sample(thk_s, np.clip(r, 0.0, nrow - 1.0),
                       np.clip(c, 0.0, ncol_s - 1.0))

    status = np.full(plon.shape, STATUS_OK, dtype=np.int8)
    status[tmin < bx.FULL_TAPER] = STATUS_FADED
    status[hit_pole] = STATUS_POLE
    status[off] = STATUS_OFFGRID
    status[hit_belt] = STATUS_BELT
    good = status == STATUS_OK

    out_thk = np.where(good, departed, np.nan)
    dthk = out_thk - np.where(good, arrive_thk, np.nan)
    # 'boxes' and 'boluses' are the same number by construction -- one box
    # carries exactly one bolus -- and both names are kept because callers
    # mean different things by them: a box is a thing on the chart you can
    # point at, a bolus is the amount it delivered.
    out = {'thk': out_thk, 'dthk': dthk, 'dtemp': dthk * K_PER_DAM,
           'boxes': dthk / float(dthk_dam),
           'boluses': dthk / float(dthk_dam), 'status': status,
           'taper': tmin, 'back_lon': plon, 'back_lat': plat,
           'substeps': substeps, 'hours': float(hours)}
    if path is not None:
        out['path_lon'] = np.array([p[0] for p in path])
        out['path_lat'] = np.array([p[1] for p in path])
    if a_sum is not None:
        with np.errstate(invalid='ignore', divide='ignore'):
            out['along_mean'] = a_sum / a_n
        out['along_frac'] = a_n / float(substeps + 1)
    return out


def trajectories(lon2d, lat2d, mslp_hpa, thk_dam, hours, *, stride=14,
                 min_boxes=0.0, **kw):
    """Back-trajectories on a lattice, as drawable paths with their box counts.

    THE thing this module measures well, in the form a chart can use: each
    parcel's path down its own isobar channel, and how many thickness intervals
    the flow carried it across on the way. Not a forecast of anything -- see
    the module header -- a measurement of transport, and one a forecaster can
    check by counting contour crossings along the drawn line.

    ``stride`` is the lattice spacing in grid cells; ``min_boxes`` drops the
    parcels that barely moved across the thickness field, which on a real chart
    is most of them and all of the clutter. Withheld parcels are dropped
    outright -- a trajectory the premise does not support is exactly the line
    that should not be drawn on a map somebody is going to believe.

    Returns a list of dicts, warmest-advection first:
        lon, lat   float64 (N,)  the path, departure point FIRST
        boxes      float        thickness intervals crossed, warm positive
        boluses    float        the same, under the quantum's own name
        dtemp      float        the same, as kelvin of layer-mean temperature
        taper      float        the worst premise taper met on the way
    """
    lon = np.asarray(lon2d, dtype='float64')
    lat = np.asarray(lat2d, dtype='float64')
    st = max(int(stride), 1)
    # Half a stride in from the frame: a lattice that starts in the corner
    # puts its first parcel where a trajectory has nowhere to come from.
    sl = (slice(st // 2, None, st), slice(st // 2, None, st))
    fc = advance(lon2d, lat2d, mslp_hpa, thk_dam, hours,
                 seeds=(lon[sl], lat[sl]), keep_path=True, **kw)
    plon, plat = fc['path_lon'], fc['path_lat']
    keep = (fc['status'] == STATUS_OK) & (np.abs(fc['boxes']) >= float(min_boxes))
    out = []
    for idx in zip(*np.nonzero(keep)):
        sel = (slice(None),) + idx
        # Reversed, so the path reads UPSTREAM-to-here and an arrowhead on the
        # last vertex points the way the air is actually going.
        out.append({'lon': plon[sel][::-1], 'lat': plat[sel][::-1],
                    'boxes': float(fc['boxes'][idx]),
                    # The same number under the quantum's name, carried here
                    # too: advance() grew 'boluses' and this did not, so the
                    # globe builder's trails came back as a KeyError that its
                    # own soft-fail swallowed into "trails skipped".
                    'boluses': float(fc['boxes'][idx]),
                    'dtemp': float(fc['dtemp'][idx]),
                    'taper': float(fc['taper'][idx])})
    out.sort(key=lambda t: -t['boxes'])
    return out


def line_forecast(lon2d, lat2d, mslp_hpa, thk_dam, leads=(6, 12, 24), *,
                  levels=CANON_LINES, **kw):
    """Where pure advection would carry the canonical thickness lines.

    NOT A PRODUCT, AND NOT TO BE MADE ONE. This is the 540-line forecast the
    module header talks somebody out of building, kept because it is the
    cleanest way to SEE the negative result -- lay these lines beside the
    verifying analysis and the overshoot is a picture instead of a table. The
    advection that draws them is three times the change the atmosphere actually
    makes (see the header), so a 540 line from here is displaced roughly three
    times too far downstream, and on a rain/snow call that is the wrong side of
    a city. Nothing in the suite draws it.

    Returns {'levels', 'analysis', 'leads': [{'hours', 'lines'}, ...]} where
    every 'lines' entry is {'dam', 'pts'} and pts is an (N, 2) lon/lat list --
    the same shape fb_meso_global ships its contour families in, so a
    verification page needs no new reader.

    Contoured only where the trajectory carried an answer. A NaN hole where the
    air came from off the chart is left as a hole rather than filled, because a
    540 line closed across ground the technique declined is the one output that
    could mislead somebody into a snow call on top of everything else.
    """
    import contourpy
    lon = np.asarray(lon2d, dtype='float64')
    lat = np.asarray(lat2d, dtype='float64')

    def _lines(field, want):
        gen = contourpy.contour_generator(
            lon, lat, np.ma.masked_invalid(np.asarray(field, dtype='float64')))
        out = []
        for lvl in want:
            for seg in gen.lines(float(lvl)):
                seg = np.asarray(seg)
                if seg.shape[0] >= 4:
                    out.append({'dam': float(lvl), 'pts': seg.tolist()})
        return out

    res = {'levels': [float(v) for v in levels],
           'analysis': _lines(thk_dam, levels), 'leads': []}
    for h in leads:
        fc = advance(lon2d, lat2d, mslp_hpa, thk_dam, h, **kw)
        res['leads'].append({'hours': float(h), 'lines': _lines(fc['thk'], levels),
                             'coverage': float(np.mean(fc['status'] == STATUS_OK))})
    return res


def skill(forecast, truth, persistence, weights=None):
    """Score a boxology forecast the only way that means anything: against
    doing nothing.

    A thickness forecast that beats no baseline is a picture. Persistence --
    the analysis itself, carried forward unchanged -- is the baseline a
    forecaster is actually entitled to for free, so the number reported is the
    Murphy skill score against it,

        SS = 1 - MSE(forecast) / MSE(persistence)

    positive meaning the operator earned its keep, zero meaning it did no
    better than the chart it started from, negative meaning it made things
    worse. Scored on the points where all three fields are finite, which is
    the honest common ground -- crediting the operator for the points it
    declined to forecast would let it win by staying quiet.

    Returns a dict: n, rmse, rmse_persistence, ss, bias, mae.
    """
    f = np.asarray(forecast, dtype='float64')
    t = np.asarray(truth, dtype='float64')
    p = np.asarray(persistence, dtype='float64')
    ok = np.isfinite(f) & np.isfinite(t) & np.isfinite(p)
    if weights is not None:
        w = np.asarray(weights, dtype='float64')
        ok &= np.isfinite(w) & (w > 0)
    if not ok.any():
        return {'n': 0, 'rmse': np.nan, 'rmse_persistence': np.nan,
                'ss': np.nan, 'bias': np.nan, 'mae': np.nan}
    w = (np.asarray(weights, dtype='float64')[ok] if weights is not None
         else np.ones(int(ok.sum())))
    ef, ep = f[ok] - t[ok], p[ok] - t[ok]
    msef = float(np.sum(w * ef * ef) / np.sum(w))
    msep = float(np.sum(w * ep * ep) / np.sum(w))
    return {'n': int(ok.sum()), 'rmse': float(np.sqrt(msef)),
            'rmse_persistence': float(np.sqrt(msep)),
            'ss': float(1.0 - msef / msep) if msep > 0 else np.nan,
            'bias': float(np.sum(w * ef) / np.sum(w)),
            'mae': float(np.sum(w * np.abs(ef)) / np.sum(w))}


def cos_weights(lat2d):
    """cos(lat) area weights, so a global score is not decided by the Arctic.

    A 0.25-deg mesh puts as many nodes in the last ten degrees of latitude as
    in the first sixty, and an unweighted global RMSE is therefore mostly a
    polar RMSE. Every score in boxology_verify.py carries these.
    """
    return np.clip(np.cos(np.radians(np.asarray(lat2d, dtype='float64'))), 0.0, None)


def box_count_summary(fc, bands=(0.5, 1.0, 2.0, 4.0)):
    """How much of the chart the forecast moves, counted in boxes.

    The readout line under the product: the fraction of forecast area that has
    crossed at least half a box, one box, two, four. Boxes rather than kelvin
    because a box is what the reader can verify by counting crossings.
    """
    b = np.asarray(fc['boxes'], dtype='float64')
    ok = np.isfinite(b)
    n = int(ok.sum())
    out = {'n': n, 'coverage': float(np.mean(fc['status'] == STATUS_OK))}
    if not n:
        return out
    a = np.abs(b[ok])
    out['bands'] = {float(t): float(np.mean(a >= t)) for t in bands}
    out['warm_frac'] = float(np.mean(b[ok] > 0))
    out['p99_boxes'] = float(np.percentile(a, 99))
    return out


def status_tally(fc):
    """Why the chart is not forecast where it is not forecast -- the same
    accounting fb_boxology's WHY_TEXT gives the shading, for the trajectory."""
    s = np.asarray(fc['status'])
    return {STATUS_TEXT[k]: float(np.mean(s == k)) for k in sorted(STATUS_TEXT)}


def _selftest():                                              # pragma: no cover
    """Consistency, on a patch where the answer is arithmetic.

    Run as `python fb_boxfcst.py`: a short step of the trajectory operator has
    to reproduce the pointwise tendency times the step, or the two halves of
    this module are not the same physics.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    lon, lat = np.meshgrid(np.linspace(-20.0, 20.0, 161),
                           np.linspace(35.0, 55.0, 81))
    # Linear fields: fb_boxology's own closed-form patch, southerly flow under
    # a warm-south thickness gradient -- textbook warm advection.
    mslp = 1000.0 + 4.0 * (lon + 20.0) / 3.0
    thk = 540.0 - 6.0 * (lat - 45.0) / 3.0
    tend = tendency(lon, lat, mslp, thk, mslp_sigma=0.0)
    fc = advance(lon, lat, mslp, thk, 1.0, mslp_sigma=0.0, substeps=64)
    mid = (slice(20, 61), slice(40, 121))
    a = float(np.nanmean(tend[mid]))
    b = float(np.nanmean(fc['dthk'][mid]))
    logging.info("tendency  %+.4f dam/h   trajectory  %+.4f dam/h   ratio %.4f",
                 a, b, b / a)
    logging.info("boxes at +1 h %+.4f   K %+.3f", float(np.nanmean(fc['boxes'][mid])),
                 float(np.nanmean(fc['dtemp'][mid])))
    logging.info("status: %s", {k: round(v, 4) for k, v in status_tally(fc).items()
                                if v})
    assert abs(b / a - 1.0) < 0.02, "trajectory and tendency disagree"
    logging.info("OK")


if __name__ == "__main__":                                    # pragma: no cover
    _selftest()
