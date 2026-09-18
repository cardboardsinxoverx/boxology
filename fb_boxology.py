"""Boxology — thermal advection read off the isobar/thickness crossings.

WHAT IT IS. On a MSLP + 1000-500 hPa thickness chart the two contour families
cut the map into quadrilaterals. Where they cross at a wide angle the boxes are
small, and the layer is being warmed or cooled hard; where they run parallel
there are no boxes and nothing is happening. Hand analysts shade those regions
red and blue by eye. This module does the same integral numerically and shades
it, so the chart carries the answer the technique is reaching for instead of
leaving every reader to do it in their head.

WHY IT IS EXACT, NOT A RULE OF THUMB. Split the layer-mean geostrophic wind
into the bottom-surface wind plus half the thermal wind, V = V0 + Vt/2. The
thermal wind blows ALONG the thickness contours, so Vt . grad(h) is identically
zero and

    -V . grad(h)  =  -V0 . grad(h)

exactly. Only the isobar-parallel flow advects thickness -- which is why the
technique is allowed to ignore everything on the chart except the two contour
families, and why it is not an approximation. Substituting the geostrophic
relation V0 = (g/f) k x grad(Z0) turns the advection into a Jacobian of the two
PLOTTED fields,

    A  =  -(g/f) J(Z0, h)  =  -(g/f) |grad Z0| |grad h| sin(theta)

with theta the crossing angle. Parallel contours give zero; perpendicular give
the maximum. Nothing else in the atmosphere enters.

WHY BOX AREA IS THE STRENGTH. Change variables from (x, y) to (Z0, h): the
Jacobian of that map IS J, so |J| dx dy = dZ0 dh, and one box bounded by two
consecutive isobars and two consecutive thickness lines integrates to

    integral over box of A dA  =  -/+ (g/f) dZ0 dh

the same magnitude for every box no matter its size or shape. Box COUNT carries
the integral; box AREA carries the local intensity, because area and |J| are
reciprocal: A_box = dZ0 dh / |J|. That reciprocity is the whole reason a
forecaster can integrate this by eye, and it is why the alpha ramp below is
keyed to intensity -- shading darkens exactly as the boxes shrink.

Converted to temperature through the hypsometric relation h = (R/g) T ln 2,

    dT/dt  =  -(g^2 / (f R ln 2)) J(Z0, h)          [K/s]

which is what advection_field() returns (in K/day).

FOUR THINGS THIS IS DELIBERATELY SHAPED AROUND.

  * SHADE WHAT IS DRAWN. J is a product of gradients, so it amplifies every
    wiggle the eye never sees -- and shading that disagrees with the visible
    crossing angles reads as a broken chart, not as extra information. Both
    input fields are therefore smoothed here to match what the chart's own
    contour calls smooth (draw_isobars uses sigma=3.0 internally; the pipeline
    hands us thickness already at sigma=1.5), and the defaults below are those
    numbers rather than whatever makes the field prettiest.
  * GEOSTROPHY HAS A LATITUDE FLOOR. 1/f runs away at the equator, so a
    tropical chart would shade a band of nonsense across its middle. The field
    is faded out between LAT_FADE_LO and LAT_FADE_HI rather than cut at a
    threshold, because a hard cut draws a straight line across the map that
    looks like an analysis boundary.
  * GEOSTROPHY ALSO FAILS IN STRONGLY CURVED FLOW. Around a tight low the
    real wind is subgeostrophic (super- around a sharp ridge), and the
    identity above is built on the geostrophic wind alone. Where the
    curvature Rossby number Ro = (g/f^2) |grad Z0| |kappa| climbs past
    RO_FADE_LO the field fades, gone by RO_FADE_HI. It FADES -- it is never
    rescaled by the gradient-wind ratio, because that ratio corrects a wind
    speed, not the Jacobian of the two plotted fields, and applying it here
    would sharpen ridge-side shading the identity has no claim to. Where the
    premise fails the honest output is silence, not a corrected guess.
  * THE LAYER NEEDS GROUND NEAR SEA LEVEL. Over high terrain the bottom of
    the 1000-500 hPa column is below the dirt, and both plotted fields are
    reductions -- conventions, not measurements (the suite already refuses
    METAR sea-level pressure above the same height, weather_maps.
    _METAR_MSLP_MAX_ELEV_M). The field fades from TERRAIN_FADE_LO, gone by
    TERRAIN_FADE_HI. Elevation is the caller's to supply (``elev_m``);
    sample_dem() reads ETOPO1's ICE SURFACE -- not the hillshade's DEM,
    which is bedrock under the Antarctic ice and puts Dome C near sea level
    beneath 3 km of it -- cached forever per extent, and a missing DEM just
    skips this taper, never the chart. The flat chart and the globe both
    take it from sample_dem, so they fade over the same ground.
"""
import logging

import cartopy.crs as ccrs
import numpy as np
from scipy.ndimage import gaussian_filter

# Physical constants (suite values; see docs/theory ch. 21).
OMEGA = 7.2921e-5        # Earth rotation rate, rad/s
G = 9.80665              # standard gravity, m/s^2
R_D = 287.058            # dry-air gas constant, J/kg/K
LN2 = np.log(2.0)        # ln(1000/500) -- the hypsometric factor for THIS layer
EARTH_R = 6.371e6        # mean Earth radius, m

# hPa -> geopotential metres at the 1000 hPa surface. The INVERSE of the
# pipeline's own mslp = 1000 + Z1000/8, so the boxology reads the same field
# the isobars were drawn from even on charts that carry true MSLP instead.
HPA_TO_GPM = 8.0

# g^2/(R ln2) ~ 0.4833 K/s^2; it is K/s per unit Jacobian only after the 1/f.
_K = G * G / (R_D * LN2)

# Colours: the suite's warm/cold pair, same red and blue as the H/L stamps and
# the thickness line families, so the chart has one colour language.
WARM_COLOR = '#FF4536'   # warm advection
COLD_COLOR = '#2E7DFF'   # cold advection

# Bands of |advection| in K/day -> (fill alpha, hatch density). Edges DOUBLE
# each step (3, 6, 12, ... 384), which is a geometric ladder in box side too
# (side ~ 1/sqrt(rate)) -- each band is the same VISUAL step down in box size,
# right down to a ~43 km box at the top end. The floor is the point where the
# technique stops having anything to say: at 50 deg, 3 K/day is a 490 km box,
# a region a hand analyst would walk past. Set it lower and two thirds of a
# mid-latitude chart shades.
#
# The FILL climbs from barely-there to a real stain as the boxes shrink --
# "deeper" is the point, this is where a tight knot of crossings should read
# as unmistakably intense, not just tinted. Hatch DENSITY climbs alongside it
# so the strength survives a grayscale print, and the two families LEAN
# OPPOSITE WAYS so the sign survives with it.
BANDS = [
    (3.0,     6.0, 0.04, 1),   # ~490-346 km -- background, barely a stain
    (6.0,    12.0, 0.07, 1),   # ~346-245 km
    (12.0,   24.0, 0.11, 2),   # ~245-173 km
    (24.0,   48.0, 0.16, 2),   # ~173-122 km
    (48.0,   96.0, 0.22, 3),   # ~122-87 km
    (96.0,  192.0, 0.29, 4),   # ~87-61 km
    (192.0, 384.0, 0.37, 5),   # ~61-43 km
    (384.0, np.inf, 0.46, 6),  # <43 km -- a knot tight enough to name
]
# Hatch stroke opacity across the same bands -- the ramp the eye actually
# reads. Kept separate from the fill alpha because contourf applies one alpha
# to face and edge together, and a stroke faint enough to be a tasteful fill is
# invisible as a line.
HATCH_ALPHA = [0.20, 0.28, 0.37, 0.46, 0.56, 0.67, 0.80, 0.95]

# Fill opacity for the BOX shading (shade_boxes), which is a different job
# from the hatch overlay's fill. There the tint only had to outline a region
# the hatching was already marking, so it stayed near-invisible; here the
# colour IS the answer, and the fill in BANDS was far too faint to read over
# the thickness ramp -- a shaded CONUS chart came out looking unshaded.
# Ends at 0.82 rather than 1.0 so the isobars, thickness lines and precip
# stay legible THROUGH the deepest box; opaque boxes would win an argument
# with the chart they are annotating.
FILL_ALPHA = [0.22, 0.31, 0.40, 0.49, 0.58, 0.67, 0.75, 0.82]
WARM_HATCH, COLD_HATCH = '/', '\\'
HATCH_LW = 0.55          # matplotlib's default hatch line is too fine to read

# Equator fade (degrees). Nothing below LO, full strength above HI.
LAT_FADE_LO, LAT_FADE_HI = 12.0, 20.0

# Curvature fade (dimensionless). Full strength while the curvature Rossby
# number Ro = (g/f^2) |grad Z0| |kappa| stays under LO, nothing past HI --
# by Ro ~ 3 the gradient-wind correction to the real wind is order-one and
# the geostrophic identity underneath has stopped describing the layer.
RO_FADE_LO, RO_FADE_HI = 1.0, 3.0

# Terrain fade (metres). LO is the suite's own line for "the sea-level
# reduction is no longer a measurement" (METAR MSLP is refused above 1000 m
# for the same reason); by HI -- the high intermountain basins and the
# Colorado Plateau floor -- the 1000 hPa surface is a few hundred hPa under
# the ground and both contour families are pure convention.
TERRAIN_FADE_LO, TERRAIN_FADE_HI = 1000.0, 2000.0

# Contour intervals the technique is being read against, for the box-size
# diagnostic only -- they do NOT enter the advection itself.
DEF_DP_HPA = 4.0         # isobar interval on the house surface charts
DEF_DTHK_DAM = 6.0       # thickness interval on the house thickness charts

ZORDER = 1.75            # over the thickness fill (1.2) and precip (1.6),
                         # under the contour lines (2.2) and the H/L stamps

# HOURS PER BOX, the readout's number. The two isobars around a box are
# geostrophic streamlines, so the flow between them carries a FIXED transport,
# g dZ0 / f per unit depth; a box is the stretch of that channel between two
# thickness lines; and the box's area over that transport is how long the flow
# takes to carry one thickness interval past a point:
#
#     hours per box  =  f A_box / (g dZ0)  =  24 dT_line / rate
#
# dT_line = (g / (R ln2)) dh is the layer-mean temperature one thickness
# interval stands for (2.96 K for the house 6 dam). Box COUNT carries the
# integral; box AREA, read this way, is a clock.
#
# A shaded box gets NO number in two places, and the readout says why:
#   * the premise is fading in it (box-mean taper under FULL_TAPER). The shade
#     dims there, and a time stretched by that dimming would be a corrected
#     guess -- the one thing the tapers exist to refuse (Evan, 2026-09-16:
#     "No number in fades").
#   * the grid cannot measure it (under MIN_TIMED_CELLS cells). A box's area
#     is a count of the cells it covers, so a small box's area -- and its
#     time -- is set by where the grid points happened to fall. Measured
#     against the box-mean pointwise field on 70,000 synthetic boxes on the
#     globe's 0.25-deg grid, the 90th-percentile scatter was 60% at 4-6
#     cells, 36% at 6-9, 27% at 9-16, 14% at 16-36 and 11% beyond. Evan,
#     2026-09-16: 16 cells. On that day's globe 0.6% of the shaded area read
#     "below grid scale" -- the knots under ~90 km a 0.25-deg grid cannot
#     time -- and the fastest box that kept a number took 13 minutes.
FULL_TAPER = 0.99
MIN_TIMED_CELLS = 16
(WHY_TIMED, WHY_SUBGRID, WHY_LATITUDE, WHY_CURVATURE, WHY_TERRAIN,
 WHY_PARTIAL) = range(6)
WHY_TEXT = {WHY_SUBGRID: 'below grid scale',
            WHY_LATITUDE: 'faded (low latitude)',
            WHY_CURVATURE: 'faded (curved flow)',
            WHY_TERRAIN: 'faded (high ground)',
            WHY_PARTIAL: 'not a whole box'}

# ...and on the wire, ONE byte per node beside the band byte:
#   0                   no box here, or no time (the band byte says which)
#   1..TIME_STEPS       hours per box, log-spaced TIME_LO_H..TIME_HI_H
#   TIME_STEPS + why    a shaded box with no time, and why (WHY_TEXT)
# Log steps because the ladder is geometric; 250 of them from a minute to two
# days is 3.2% a step, finer than the readout rounds to.
TIME_STEPS = 250
TIME_LO_H, TIME_HI_H = 1.0 / 60.0, 48.0


def _metric(lon2d, lat2d):
    """dx, dy in metres for a lon/lat mesh (lon along axis 1, lat along axis 0
    -- the suite convention, same as the vorticity/divergence operators).

    Longitude is unwrapped before differencing: on a global mesh the 0/360 seam
    otherwise puts a 360-degree jump in one column, which becomes a near-zero
    dx and a spike of advection down the edge of the map."""
    lon = np.unwrap(np.radians(np.asarray(lon2d, dtype='float64')), axis=1)
    lat = np.radians(np.asarray(lat2d, dtype='float64'))
    dx = EARTH_R * np.cos(lat) * np.gradient(lon, axis=1)
    dy = EARTH_R * np.gradient(lat, axis=0)
    # A degenerate row/column (repeated coordinate) would divide by zero.
    dx = np.where(np.abs(dx) < 1.0, np.nan, dx)
    dy = np.where(np.abs(dy) < 1.0, np.nan, dy)
    return dx, dy


def _curvature_taper(dzdx, dzdy, dx, dy, f):
    """1 where the flow is straight, 0 where it is too curved to trust.

    kappa is the signed curvature of the Z0 contours -- the geostrophic
    streamlines -- so Ro = V |kappa| / f = (g/f^2) |grad Z0| |kappa| without
    ever forming the wind. A REDUCTION only, clipped to [0, 1]: see the
    module docstring for why the gradient-wind ratio must not appear here."""
    zxx = np.gradient(dzdx, axis=1) / dx
    zyy = np.gradient(dzdy, axis=0) / dy
    zxy = np.gradient(dzdx, axis=0) / dy
    gmag2 = dzdx * dzdx + dzdy * dzdy
    # |kappa| = |Zxx Zy^2 - 2 Zxy Zx Zy + Zyy Zx^2| / |grad Z|^3, and the
    # |grad Z0| factor of Ro cancels one power, so |grad Z|^2 divides out.
    num = np.abs(zxx * dzdy * dzdy - 2.0 * zxy * dzdx * dzdy
                 + zyy * dzdx * dzdx)
    with np.errstate(divide='ignore', invalid='ignore'):
        ro = (G / (f * f)) * num / gmag2
    # A col or ridge axis (grad Z -> 0) sends kappa to infinity, but calm air
    # is not curved FLOW -- and the Jacobian is already zero there, so calling
    # it straight tapers nothing that was going to draw.
    ro = np.where(gmag2 < 1e-12, 0.0, ro)
    return np.clip((RO_FADE_HI - ro) / (RO_FADE_HI - RO_FADE_LO), 0.0, 1.0)


def _terrain_taper(elev_m):
    """1 near sea level, 0 above the band where the layer is fiction."""
    e = np.asarray(elev_m, dtype='float64')
    return np.clip((TERRAIN_FADE_HI - e) / (TERRAIN_FADE_HI - TERRAIN_FADE_LO),
                   0.0, 1.0)


def sample_dem(lon2d, lat2d):
    """Surface elevation (m) on the chart's own mesh, or None.

    The ICE surface (fb_topo.get_surface_elevation, ETOPO1), never the
    hillshade's DEM: the taper asks how high the air's floor is, and over
    Antarctica the hillshade's raster answers with the rock under the ice.
    Cached forever per mesh box, so a chart pays one fetch per domain, once.
    A mesh that goes all the way round (the globe's, seam padded to
    -180.5..180.25) reads one whole-world raster and wraps into it.
    Nearest-neighbour is plenty against a 1000 m-wide fade.
    Returns None on any failure (offline first render, NCEI down, fb_topo
    missing); callers hand the result to ``elev_m``, where None simply skips
    the terrain taper -- a chart is never held hostage to a topo fetch."""
    try:
        import fb_topo
        lon = np.asarray(lon2d, dtype='float64')
        lat = np.asarray(lat2d, dtype='float64')
        extent = [float(np.nanmin(lon)), float(np.nanmax(lon)),
                  float(np.nanmin(lat)), float(np.nanmax(lat))]
        dem = fb_topo.get_surface_elevation(extent)
        if dem is None:
            return None
        # Index math must use the SAME box the raster was keyed to, so
        # normalize exactly as fb_topo does (0.25-deg quantized, e > 180
        # flagging a dateline crossing, a full circle folded to -180..180).
        # Row 0 of the DEM is the northern edge.
        w, e, s, n = fb_topo._norm_extent(extent)
        hgt, wid = dem.shape
        lon_u = np.mod(lon - w, 360.0) + w
        col = np.clip(np.rint((lon_u - w) / (e - w) * (wid - 1)),
                      0, wid - 1).astype(np.intp)
        row = np.clip(np.rint((n - lat) / (n - s) * (hgt - 1)),
                      0, hgt - 1).astype(np.intp)
        return dem[row, col]
    except Exception as exc:                                 # noqa: BLE001
        logging.info(f"Boxology: DEM unavailable, terrain taper skipped ({exc}).")
        return None


def _advection_parts(lon2d, lat2d, mslp_hpa, thk_dam,
                     mslp_sigma=3.0, thk_sigma=0.0,
                     curvature_taper=True, elev_m=None):
    """The shared core: returns (advection masked array, taper field, belt,
    parts).

    The taper field is the product of every premise guard -- latitude,
    curvature, terrain -- and is what box_shading averages per box so the
    box rate fades exactly where the pointwise field does. ``parts`` holds
    each guard's own factor (None for one that did not run), so a readout
    can say WHICH premise is failing where a box has faded."""
    lat = np.asarray(lat2d, dtype='float64')
    p = np.asarray(mslp_hpa, dtype='float64')
    h = np.asarray(thk_dam, dtype='float64') * 10.0        # dam -> m

    if mslp_sigma:
        p = gaussian_filter(p, sigma=mslp_sigma)
    if thk_sigma:
        h = gaussian_filter(h, sigma=thk_sigma)

    z0 = (p - 1000.0) * HPA_TO_GPM                          # 1000 hPa height, m

    dx, dy = _metric(lon2d, lat)
    dzdx = np.gradient(z0, axis=1) / dx
    dzdy = np.gradient(z0, axis=0) / dy
    dhdx = np.gradient(h, axis=1) / dx
    dhdy = np.gradient(h, axis=0) / dy

    jac = dzdx * dhdy - dzdy * dhdx                         # dimensionless

    # Equator fade: taper to zero through the belt rather than cutting a line
    # across the chart, and drop the belt itself, where 1/f is meaningless.
    #
    # The belt is excluded from the DIVISION, not just from the result. Dividing
    # first and masking after produces inf inside the belt, and inf * 0 (the
    # taper's own zero) is nan, not zero -- so every chart reaching the tropics
    # raised "invalid value encountered in multiply" on its way to an answer
    # that was going to be masked regardless.
    a = np.abs(lat)
    belt = a < LAT_FADE_LO
    f = 2.0 * OMEGA * np.sin(np.radians(lat))
    f = np.where(belt, np.nan, f)
    taper = np.clip((a - LAT_FADE_LO) / (LAT_FADE_HI - LAT_FADE_LO), 0.0, 1.0)
    parts = {'latitude': taper, 'curvature': None, 'terrain': None}
    if curvature_taper:
        parts['curvature'] = _curvature_taper(dzdx, dzdy, dx, dy, f)
        taper = taper * parts['curvature']
    if elev_m is not None:
        parts['terrain'] = _terrain_taper(elev_m)
        taper = taper * parts['terrain']
    with np.errstate(divide='ignore', invalid='ignore'):
        adv = -(_K / f) * jac * taper * 86400.0             # K/day

    adv = np.ma.masked_invalid(np.ma.masked_where(belt, adv))
    return adv, taper, belt, parts


def advection_field(lon2d, lat2d, mslp_hpa, thk_dam,
                    mslp_sigma=3.0, thk_sigma=0.0,
                    curvature_taper=True, elev_m=None):
    """Layer-mean (1000-500 hPa) geostrophic temperature advection, K/day.

    Positive is warm advection. Returns a masked array -- masked over the
    equatorial belt where the geostrophic relation does not hold, and faded
    (never gradient-wind corrected) where the flow is too curved or the
    ground too high for the premise; see the module docstring.

    ``mslp_sigma`` / ``thk_sigma`` are the EXTRA smoothing applied here. The
    defaults assume the caller is the house thickness chart, which hands over
    raw MSLP (draw_isobars will smooth it at sigma=3.0 when it draws it) and
    thickness already smoothed at sigma=1.5. Match them to whatever the chart
    actually contours.

    ``curvature_taper`` is pure arithmetic and defaults on. ``elev_m`` is an
    elevation field on the same mesh (sample_dem() supplies one); None skips
    the terrain taper."""
    adv, _taper, _belt, _parts = _advection_parts(
        lon2d, lat2d, mslp_hpa, thk_dam, mslp_sigma, thk_sigma,
        curvature_taper, elev_m)
    return adv


def box_side_km(lon2d, lat2d, mslp_hpa, thk_dam,
                dp_hpa=DEF_DP_HPA, dthk_dam=DEF_DTHK_DAM, **kw):
    """Side of the equivalent square box, km -- the thing the eye is actually
    measuring. A_box = dZ0 dh / |J|, so this is just the advection field read
    back through the contour intervals; supplied for labelling and for the
    self-test, not used by the shading."""
    adv = advection_field(lon2d, lat2d, mslp_hpa, thk_dam, **kw)
    f = 2.0 * OMEGA * np.sin(np.radians(np.asarray(lat2d, dtype='float64')))
    dz = dp_hpa * HPA_TO_GPM
    dh = dthk_dam * 10.0
    with np.errstate(divide='ignore', invalid='ignore'):
        area = _K * dz * dh * 86400.0 / (np.abs(f) * np.abs(adv))
    return np.ma.masked_invalid(np.ma.sqrt(area)) / 1000.0


def label_boxes(lon2d, lat2d, mslp_hpa, thk_dam,
                dp_hpa=DEF_DP_HPA, dthk_dam=DEF_DTHK_DAM,
                mslp_sigma=3.0, thk_sigma=0.0):
    """Label the actual quadrilaterals the two contour families cut the map
    into -- the BOXES, as drawn, not a smooth field standing in for them.

    A box is one connected region over which both band indices are constant:
    floor(p / dp) picks which pair of isobars you are between, floor(h / dthk)
    which pair of thickness lines. The pair changes exactly where a contour is
    drawn, so a region of constant (iz, ih) IS the polygon bounded by the four
    lines around it. Nothing here has to trace or intersect a contour.

    4-connectivity, not 8: two boxes touching only at a corner meet AT the
    crossing point of one isobar and one thickness line, which is the one place
    they are guaranteed to be different boxes. 8-connectivity welds them into a
    single blob straddling the crossing -- exactly the feature being measured.

    Returns (labels, count), labels 0 where nothing is boxed.
    """
    from scipy.ndimage import label as _label

    p = np.asarray(mslp_hpa, dtype='float64')
    h = np.asarray(thk_dam, dtype='float64')
    if mslp_sigma:
        p = gaussian_filter(p, sigma=mslp_sigma)
    if thk_sigma:
        h = gaussian_filter(h, sigma=thk_sigma)

    iz = np.floor(p / float(dp_hpa))
    ih = np.floor(h / float(dthk_dam))
    good = np.isfinite(iz) & np.isfinite(ih)
    # One integer per (isobar band, thickness band) pair. The multiplier only
    # has to exceed the thickness-band span to keep pairs from colliding.
    pair = np.where(good, iz * 100003.0 + ih, np.nan)

    labels = np.zeros(pair.shape, dtype=np.int32)
    n = 0
    # ~10 isobar bands x ~10 thickness bands on a synoptic chart, so this loop
    # is tens of iterations, not thousands.
    for v in np.unique(pair[good]):
        lab, k = _label(pair == v)
        if k:
            labels[lab > 0] = lab[lab > 0] + n
            n += k
    return labels, n


def _whole_box_mask(labels, n, mslp_hpa, thk_dam, dp_hpa, dthk_dam,
                    mslp_sigma=3.0, thk_sigma=0.0):
    """True for each label 1..n that is a WHOLE box: it touches a region one
    isobar band up AND one down AND one thickness band up AND one down. Read
    off the labelled mesh's own 4-neighbour edges, so it needs no contour
    tracing and no span threshold that a small box's few cells would fail."""
    p = np.asarray(mslp_hpa, dtype='float64')
    h = np.asarray(thk_dam, dtype='float64')
    if mslp_sigma:
        p = gaussian_filter(p, sigma=mslp_sigma)
    if thk_sigma:
        h = gaussian_filter(h, sigma=thk_sigma)
    iz = np.floor(p / float(dp_hpa))
    ih = np.floor(h / float(dthk_dam))
    seen = np.zeros((4, n + 1), dtype=bool)      # p up, p down, h up, h down
    for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        for src, dst in ((a, b), (b, a)):
            lab = labels[src]
            dz = iz[dst] - iz[src]
            dh = ih[dst] - ih[src]
            for k, hit in enumerate((dz > 0, dz < 0, dh > 0, dh < 0)):
                seen[k, lab[hit & (lab > 0)]] = True
    return seen[:, 1:].all(axis=0)


def box_shading(lon2d, lat2d, mslp_hpa, thk_dam,
                dp_hpa=DEF_DP_HPA, dthk_dam=DEF_DTHK_DAM,
                drop_edge=True, **kw):
    """Per-box band index: 0 = unshaded, +k = warm band k, -k = cold band k.

    THE WHOLE TECHNIQUE, evaluated the way it is actually taught. Each box
    integrates to the same quantum of advection (see the module docstring), so
    a box's MEAN rate is that quantum over its own area -- big slack box, weak;
    small tight box, strong. Every cell of a box gets its box's band, so a box
    fills as one flat colour the way a hand analyst shades it, instead of the
    smooth per-pixel field a contourf of the same quantity would give.

    ``drop_edge`` discards boxes touching the frame: those are CUT boxes whose
    measured area is the part that fits on the chart, not the box's real area,
    so they would read as far tighter than they are. A hand analyst does not
    count a box running off the map either.

    The taper kwargs (``curvature_taper``, ``elev_m``) ride along to the
    advection core; the box rate is scaled by the box-MEAN taper, so a box
    fades exactly as much as the pointwise field inside it does and the two
    renderings never disagree about where the premise holds.

    This is box_fields' band; the globe, which also wants the time, calls
    box_fields and pays for the labelling once.
    """
    return box_fields(lon2d, lat2d, mslp_hpa, thk_dam, dp_hpa, dthk_dam,
                      drop_edge, **kw)['band']


def box_fields(lon2d, lat2d, mslp_hpa, thk_dam,
               dp_hpa=DEF_DP_HPA, dthk_dam=DEF_DTHK_DAM,
               drop_edge=True, **kw):
    """box_shading's band AND the readout's time, from ONE labelling pass.

    Returns a dict of per-node arrays, every cell carrying its box's values:

        band   int16    0 unshaded, +k warm band k, -k cold band k
        hours  float64  hours per box (see HOURS PER BOX), NaN where untimed
        why    int8     WHY_TIMED, or why a SHADED box carries no hours --
                        the fade that dims it most, or below grid scale

    The hours come off the same rate the band is binned from, so a number is
    never shown for a box the shade does not draw.

    ``whole_boxes`` (ON by default since 2026-09-18, Evan: "make whole_boxes
    the app default"; False is the first implementation, kept for the
    before/after). The quantum belongs to a WHOLE box: a region
    with an isobar on two sides and a thickness line on the other two. The
    labelling also returns regions that are not that -- the strip between two
    isobars inside a thickness plateau, the ring between two closed isobars
    around a low, a tongue one thickness line enters and leaves by the same
    side -- and those do not integrate to a quantum (a ring integrates to
    ZERO: Green's theorem, Z0 constant on both of its boundaries). Measured on
    the 2026-09-07 12Z Southern Hemisphere chart, whole boxes matched the
    box-mean pointwise field to a median 1.004 (10th-90th percentile
    0.97-1.06); the partial regions, a fifth of the regions and a quarter of
    the shaded area, were rated a median 177 times too high. With
    ``whole_boxes=True`` a partial region is rated at its box-mean pointwise
    |A| instead -- what a whole box's quantum over area equals anyway -- and
    carries no hours (the readout says 'not a whole box'), because it is not
    a channel segment with a transit time."""
    from scipy.ndimage import mean as _lmean, sum as _lsum

    # label_boxes cuts geometry from the two contoured fields alone -- the
    # premise tapers are advection-side arguments and must not reach it.
    curv = kw.pop('curvature_taper', True)
    elev_m = kw.pop('elev_m', None)
    whole_boxes = kw.pop('whole_boxes', True)
    labels, n = label_boxes(lon2d, lat2d, mslp_hpa, thk_dam,
                            dp_hpa, dthk_dam, **kw)
    # Per-LABEL tables, read back through `labels` at the end: one pass over
    # the mesh instead of one full-mesh comparison per shaded box. Label 0
    # (unboxed) keeps the defaults.
    band_of = np.zeros(n + 1, dtype=np.int16)
    hours_of = np.full(n + 1, np.nan)
    why_of = np.zeros(n + 1, dtype=np.int8)
    if not n:
        return {'band': band_of[labels], 'hours': hours_of[labels],
                'why': why_of[labels]}

    adv, taper, _belt, parts = _advection_parts(
        lon2d, lat2d, mslp_hpa, thk_dam, curvature_taper=curv,
        elev_m=elev_m, **kw)
    lat = np.asarray(lat2d, dtype='float64')
    dx, dy = _metric(lon2d, lat)
    cell = np.abs(dx * dy)                                  # m^2 per grid cell

    idx = np.arange(1, n + 1)
    area = np.asarray(_lsum(np.nan_to_num(cell), labels, idx))          # m^2
    signed = np.asarray(_lmean(np.nan_to_num(adv.filled(0.0)), labels, idx))
    fbar = np.asarray(_lmean(np.abs(2.0 * OMEGA * np.sin(np.radians(lat))),
                             labels, idx))
    tbar = np.asarray(_lmean(np.nan_to_num(taper), labels, idx))

    # The quantum, per box: (g^2/(R ln2)) * dZ0 * dh / f, in K.m^2/day.
    quantum = (_K * (dp_hpa * HPA_TO_GPM) * (dthk_dam * 10.0) * 86400.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = quantum * tbar / (fbar * area)               # K/day, box mean

    whole = np.ones(n, dtype=bool)
    if whole_boxes:
        whole = _whole_box_mask(labels, n, mslp_hpa, thk_dam, dp_hpa, dthk_dam,
                                kw.get('mslp_sigma', 3.0),
                                kw.get('thk_sigma', 0.0))
        rate = np.where(whole, rate, np.abs(signed))

    if drop_edge:
        edge = set(np.unique(np.concatenate([
            labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])))
        edge.discard(0)
    else:
        edge = set()

    # For the readout: each box's cell count, each guard's own box mean (the
    # lowest names the fade), and the temperature one thickness line is.
    cells = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    guard_means = [(why, np.asarray(_lmean(np.nan_to_num(parts[key]),
                                           labels, idx)))
                   for why, key in ((WHY_LATITUDE, 'latitude'),
                                    (WHY_CURVATURE, 'curvature'),
                                    (WHY_TERRAIN, 'terrain'))
                   if parts[key] is not None]
    dt_line = G / (R_D * LN2) * (dthk_dam * 10.0)          # K per interval

    lo_edges = np.array([b[0] for b in BANDS])
    for i, lab in enumerate(idx):
        r = rate[i]
        if lab in edge or not np.isfinite(r) or r < BANDS[0][0]:
            continue
        if signed[i] == 0.0:
            continue
        band = int(np.searchsorted(lo_edges, r, side='right'))   # 1..len(BANDS)
        band_of[lab] = band if signed[i] > 0 else -band
        if not whole[i]:
            why_of[lab] = WHY_PARTIAL
        elif tbar[i] < FULL_TAPER:
            why_of[lab] = min(guard_means, key=lambda wm: wm[1][i])[0]
        elif cells[i] < MIN_TIMED_CELLS:
            why_of[lab] = WHY_SUBGRID
        else:
            hours_of[lab] = 24.0 * dt_line / r
    return {'band': band_of[labels], 'hours': hours_of[labels],
            'why': why_of[labels]}


def box_time_bytes(hours, why):
    """box_fields' hours and why as the wire's one byte per node (see TIME_STEPS
    above): 0 untimed, 1..TIME_STEPS log-quantized hours, TIME_STEPS + why for
    a shaded box that carries no number."""
    h = np.asarray(hours, dtype='float64')
    w = np.asarray(why, dtype='int16')
    out = np.zeros(h.shape, dtype=np.uint8)
    timed = np.isfinite(h) & (h > 0)
    lo, hi = np.log(TIME_LO_H), np.log(TIME_HI_H)
    step = np.rint((np.log(np.clip(h[timed], TIME_LO_H, TIME_HI_H)) - lo)
                   / (hi - lo) * (TIME_STEPS - 1))
    out[timed] = (step + 1).astype(np.uint8)
    held = ~timed & (w > WHY_TIMED)
    out[held] = (TIME_STEPS + w[held]).astype(np.uint8)
    return out


def box_time_meta():
    """What the viewer needs to read box_time_bytes back: the log scale, and
    the words for every held-back code."""
    return {'steps': TIME_STEPS, 'lo_h': TIME_LO_H, 'hi_h': TIME_HI_H,
            'why': {str(TIME_STEPS + k): text for k, text in WHY_TEXT.items()}}


def shade_boxes(ax, lon2d, lat2d, mslp_hpa, thk_dam, *,
                zorder=ZORDER, transform=None, alpha_scale=1.0, **kw):
    """Fill each box solid: red warm, blue cold, deeper as the box tightens.

    Drawn as ONE pcolormesh over a band-index field rather than a polygon per
    box. pcolormesh takes the mesh as given, so it needs no assumption that the
    grid is evenly spaced, and the band edges land exactly on the contour lines
    already drawn on top of it -- which is the point: the colour has to stop
    where the isobar is, or it is not shading a box.
    """
    import matplotlib.colors as mcolors

    if transform is None:
        transform = ccrs.PlateCarree()
    band = box_shading(lon2d, lat2d, mslp_hpa, thk_dam, **kw)
    if not np.any(band):
        return None

    nb = len(BANDS)
    # Colour table indexed by band: cold -nb..-1, then warm 1..nb.
    cold = [mcolors.to_rgba(COLD_COLOR, min(1.0, FILL_ALPHA[i] * alpha_scale))
            for i in range(nb)][::-1]
    warm = [mcolors.to_rgba(WARM_COLOR, min(1.0, FILL_ALPHA[i] * alpha_scale))
            for i in range(nb)]
    cmap = mcolors.ListedColormap(cold + warm)
    # Bin edges around the integer band values, skipping 0 (unshaded).
    levels = [-nb - 0.5] + [v + 0.5 for v in range(-nb, 0)] \
             + [v + 0.5 for v in range(1, nb + 1)]
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    shown = np.ma.masked_where(band == 0, band)
    ax.pcolormesh(lon2d, lat2d, shown, cmap=cmap, norm=norm,
                  shading='nearest', zorder=zorder, transform=transform)
    return band


def shade_boxology(ax, lon2d, lat2d, mslp_hpa, thk_dam, *,
                   zorder=ZORDER, transform=None, **kw):
    """Crosshatch warm (red) and cold (blue) advection onto ``ax``.

    Returns the advection field so a caller can annotate from it. Never raises:
    a failure here drops the overlay, never the chart."""
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    if transform is None:
        transform = ccrs.PlateCarree()
    adv = advection_field(lon2d, lat2d, mslp_hpa, thk_dam, **kw)
    if adv is None or adv.count() == 0:
        return None

    prev_lw = mpl.rcParams.get('hatch.linewidth')
    mpl.rcParams['hatch.linewidth'] = HATCH_LW
    try:
        for color, hatch, sign in ((WARM_COLOR, WARM_HATCH, 1.0),
                                   (COLD_COLOR, COLD_HATCH, -1.0)):
            # Multiply through by the sign so both passes read as "bigger is
            # stronger" and the band table is written once.
            signed = adv * sign
            top = float(signed.max()) if signed.count() else 0.0
            for i, (lo, hi, alpha, dens) in enumerate(BANDS):
                if top <= lo:
                    break
                hi_eff = top + 1.0 if not np.isfinite(hi) else hi
                # Two passes. contourf applies ONE alpha to face and edge, and
                # the fill has to stay far fainter than the hatch, so the tint
                # and the strokes are drawn separately.
                ax.contourf(lon2d, lat2d, signed, levels=[lo, hi_eff],
                            colors=[color], alpha=alpha, zorder=zorder,
                            extend='neither', transform=transform)
                cs = ax.contourf(lon2d, lat2d, signed, levels=[lo, hi_eff],
                                 colors='none', hatches=[hatch * dens],
                                 zorder=zorder + 0.01, extend='neither',
                                 transform=transform)
                edge = mcolors.to_rgba(color, HATCH_ALPHA[i])
                try:
                    cs.set_edgecolor(edge)
                except Exception:
                    # Older matplotlib exposes the parts as .collections.
                    for c in getattr(cs, 'collections', []):
                        c.set_edgecolor(edge)
    except Exception as e:
        logging.warning(f"Boxology: advection shading skipped ({e}).")
        return None
    finally:
        if prev_lw is not None:
            mpl.rcParams['hatch.linewidth'] = prev_lw
    return adv


def key_handles(n=3, hatched=False):
    """Legend handles for the key band under the map: ``n`` chips per sign,
    sampled off the 8-band ladder from slack to knotted-tight, each labelled
    with the box size the threshold corresponds to at 50 deg so the reader can
    go back to counting boxes with a calibrated eye.

    ``hatched`` matches the chip to shade_boxology's crosshatch; the default
    matches shade_boxes' solid fill, which is what the chart draws.

    ``n=3`` (the default) is right for a standalone key -- e.g. the theory
    figure -- where there is room to show the ramp has real depth. The house
    thickness chart embeds this INTO the existing reference-line key band,
    which has room for one more row, not three; it calls ``n=2`` (slack and
    knotted only) to fit."""
    from matplotlib.patches import Patch

    n = max(2, min(n, len(BANDS)))
    idx = sorted({0, len(BANDS) - 1} | {round(i * (len(BANDS) - 1) / (n - 1))
                                        for i in range(n)})
    labels = {0: 'slack', len(BANDS) - 1: 'knotted'}

    def chip(color, hatch, i, sign_label):
        lo, hi, alpha, dens = BANDS[i]
        side = box_side_for_rate(lo)
        end = labels.get(i, 'closing')
        rate = f'>{lo:.0f}' if not np.isfinite(hi) else f'{lo:.0f}-{hi:.0f}'
        # The chip has to be the swatch the map actually draws, at the alpha
        # the map actually draws it: a solid chip beside crosshatched boxes
        # (or a fully opaque one beside a 0.22 fill) is a key that describes a
        # different chart.
        return Patch(facecolor=color, alpha=1.0 if hatched else FILL_ALPHA[i],
                     hatch=(hatch * dens) if hatched else None,
                     edgecolor=color,
                     label=f'{sign_label} \u2014 {end} box '
                           f'(~{side:.0f} km, {rate} K/day)')

    handles = [chip(WARM_COLOR, WARM_HATCH, i, 'warm advection') for i in idx]
    handles += [chip(COLD_COLOR, COLD_HATCH, i, 'cold advection') for i in idx]
    return handles


def box_side_for_rate(k_per_day, lat=50.0, dp_hpa=DEF_DP_HPA,
                      dthk_dam=DEF_DTHK_DAM):
    """Side of the square box that produces ``k_per_day`` at ``lat``, km. The
    inverse of the shading ramp, for labelling: it is what "this dark" means
    back in the units the technique is actually read in."""
    f = 2.0 * OMEGA * np.sin(np.radians(lat))
    area = _K * (dp_hpa * HPA_TO_GPM) * (dthk_dam * 10.0) * 86400.0 / (f * k_per_day)
    return float(np.sqrt(area)) / 1000.0
