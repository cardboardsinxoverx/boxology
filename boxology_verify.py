#!/usr/bin/env python3
"""boxology_verify.py — the boxology measured against real GFS grids.

WHY THIS EXISTS, and what it found. fb_boxology derives its advection in closed
form, and docs/theory ch. 21 asserts the derivation. Neither says whether the
shortcut survives contact with a real atmosphere, and the answer turned out to
be worth having in three separate pieces -- two of which are good news for the
technique and one of which kills an obvious product nobody should build.

  1. FIDELITY. Does reading the advection off two contour families actually
     measure the advection? Against the model's OWN layer-mean wind, on its own
     grid: correlation +0.94. The identity holds on real data. The amplitude
     runs about 1.39x strong, and that is explained rather than mysterious --
     the geostrophic wind at 1000 hPa is 7.4 m/s rms where the model's actual
     1000 hPa wind is 5.2, because the real near-surface wind is frictionally
     retarded and backed across the isobars. Notably the boxology's V0 lands
     almost exactly on the model's 850 mb wind (7.2 m/s), which is the first
     level above the friction layer -- so what the two contour families are
     really measuring is the 850 mb thermal advection, the field the chart
     already draws barbs for.

  2. ATTRIBUTION. Is that advection the local temperature tendency? No, and not
     nearly. Against a centered 6-hour thickness tendency the regression slope
     is 0.31: advection over-predicts the actual change by more than three
     times, consistently across smoothing scales from 40 to 330 km. Roughly two
     thirds of every advective signal is canceled -- adiabatic cooling by the
     ascent that the warm advection itself forces, the compensation the QG
     omega equation ties together. This is a property of the atmosphere, not a
     defect of the boxology: the model's own full-wind advection is canceled
     just as hard (slope 0.33).

     The pleasant surprise inside it: the boxology's GEOSTROPHIC advection
     tracks the net tendency BETTER than the model's total advection does
     (correlation +0.50 against +0.41). Restricting the identity to the
     geostrophic wind is not a compromise -- the ageostrophic advection it
     leaves out is disproportionately the part that gets adiabatically
     canceled, so throwing it away moves the answer closer to the truth.

  3. CONSEQUENCE. A kinematic forecast -- take the advection rate, multiply by
     a time step, call it a temperature forecast -- therefore has NEGATIVE
     skill against persistence at every lead tested (SS -0.74 at 6 h, -0.87 at
     12 h, -2.3 at 24 h). It is not a small error; it is a forecast three times
     too large. This harness prints that row on purpose, so the next person to
     have the idea can read the number instead of rediscovering it.

     Scaled by the measured compensation the same operator turns slightly
     positive (SS about +0.12 at 6 h), and that scaling is NOT applied anywhere
     in the suite. A coefficient regressed against GFS is exactly the corrected
     guess fb_boxology refuses when it fades rather than applying a
     gradient-wind ratio; it is reported here as a measurement and left out of
     the module.

HOW IT IS SCORED. On the points where the technique agreed to speak --
fb_boxfcst withholds an answer wherever the premise fails along a trajectory --
with coverage printed beside every number, because a score on 70% of a window
is a different claim from a score on all of it. Weighted by cos(lat), or the
poleward rows decide a continental score. The tendency is a CENTERED difference
about the analysis hour, not a one-sided one: a forward difference charges the
advection for half a step of pattern motion it never claimed.

    python boxology_verify.py                       # the whole report
    python boxology_verify.py --window europe
    python boxology_verify.py --skip-skill --json out.json
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.ndimage import gaussian_filter

import fb_boxfcst as bf
import fb_boxology as bx
import fb_gfsarchive as ga
import fb_gfsbox as gb

log = logging.getLogger("boxology_verify")

# The fetch box is deliberately much bigger than the scoring window. A 24-hour
# trajectory at 30 m/s reaches back 2,600 km, so a window scored on its own
# extent would have its whole upwind third flagged "air came from off the
# chart" -- and the operator would then be scored only where the flow was slow,
# which is exactly where advection has least to say.
WINDOWS = {
    'conus': {'fetch': (-170.0, -40.0, 12.0, 72.0),
              'score': (-125.0, -67.0, 25.0, 52.0)},
    'europe': {'fetch': (-70.0, 60.0, 20.0, 75.0),
               'score': (-12.0, 32.0, 36.0, 62.0)},
    'pacific': {'fetch': (110.0, 240.0, 12.0, 72.0),
                'score': (150.0, 205.0, 25.0, 55.0)},
    # Southern Hemisphere, where f changes sign and cold air comes from the
    # other side of the map. Mostly open ocean: no terrain taper to lean on,
    # and no radiosonde network propping the analysis up either.
    'sindian': {'fetch': (-20.0, 170.0, -78.0, -8.0),
                'score': (30.0, 115.0, -60.0, -28.0)},
    'satlantic': {'fetch': (-120.0, 60.0, -78.0, -8.0),
                  'score': (-50.0, 15.0, -60.0, -28.0)},
}

LEVELS = (1000, 850, 700, 500)
Z_FIELDS = [gb.isobaric('z500', 'HGT', 'gh', 500),
            gb.isobaric('z1000', 'HGT', 'gh', 1000)]
WIND_FIELDS = Z_FIELDS + [gb.isobaric(f'{c}{lev}', v, c, lev)
                          for lev in LEVELS
                          for c, v in (('u', 'UGRD'), ('v', 'VGRD'))]

# Model omega, for the "is warm advection a lift zone?" question. Pa/s.
OMEGA_LEVELS = (850, 700, 500)
OMEGA_FIELDS = [gb.isobaric(f'w{lev}', 'VVEL', 'w', lev) for lev in OMEGA_LEVELS]

# The chart's own recipe, so the technique is measured on the fields a
# forecaster is actually looking at: fb_meso_global.build_thickness_tile's
# thickness (smoothed 1.5 sigma) and weather_maps' MSLP (1000 + Z1000/8).
THK_SIGMA = 1.5
MSLP_SIGMA = 3.0

# The centered tendency's half-window, hours. Six hours centered on the
# analysis is the shortest span GFS posts on both sides at every cycle, and
# short enough that the pattern has not reorganized inside it.
TEND_HALF_H = 3


def _thk(box, sigma=THK_SIGMA):
    return gaussian_filter((np.asarray(box['z500']) - np.asarray(box['z1000']))
                           / 10.0, sigma=sigma)


def _mslp(box):
    return 1000.0 + np.asarray(box['z1000']) / 8.0


def _mesh(box):
    return np.meshgrid(np.asarray(box['lon']), np.asarray(box['lat']))


def _fetch(extent, fields, tag, fhr=0, cycle=None):
    """One door for both data roads. A pinned `cycle` reads NOAA's open-data
    archive (fb_gfsarchive: any cycle since 2021, and the only road to a
    season other than this one); no cycle reads the newest run off NOMADS."""
    if cycle is not None:
        return ga.fetch(extent, fields, tag, fhr=fhr, cycle=cycle)
    return gb.fetch(extent, fields, tag, fhr=fhr, pad=0.0)


def _grab(extent, fhr, tag, cycle=None):
    box = _fetch(extent, Z_FIELDS, tag, fhr=fhr, cycle=cycle)
    if box is None:
        raise SystemExit(f"boxology_verify: GFS f{fhr:03d} unavailable; "
                         "try again shortly.")
    return box


def _window_mask(lon2d, lat2d, score):
    lo0, lo1, la0, la1 = score
    lon = np.mod(lon2d - lo0, 360.0) + lo0
    return (lon >= lo0) & (lon <= lo1) & (lat2d >= la0) & (lat2d <= la1)


class _Stats:
    """Weighted correlation, regression slope and rms over one fixed mask.

    One object per comparison region so every number in the report is taken on
    exactly the same points -- the alternative is a table whose rows quietly
    disagree about which grid cells they describe.
    """

    def __init__(self, weights, mask):
        self.w = np.asarray(weights, dtype='float64') * mask

    def pair(self, a, b):
        a = np.asarray(a, dtype='float64')
        b = np.asarray(b, dtype='float64')
        m = (self.w > 0) & np.isfinite(a) & np.isfinite(b)
        if not m.any():
            return {'corr': np.nan, 'slope': np.nan, 'rms_x': np.nan,
                    'rms_y': np.nan, 'n': 0}
        w = self.w[m]
        sw = np.sum(w)
        x, y = a[m], b[m]
        mx, my = np.sum(w * x) / sw, np.sum(w * y) / sw
        vx = np.sum(w * (x - mx) ** 2) / sw
        vy = np.sum(w * (y - my) ** 2) / sw
        cv = np.sum(w * (x - mx) * (y - my)) / sw
        return {'corr': float(cv / np.sqrt(vx * vy)) if vx * vy > 0 else np.nan,
                'slope': float(cv / vx) if vx > 0 else np.nan,
                'rms_x': float(np.sqrt(np.sum(w * x * x) / sw)),
                'rms_y': float(np.sqrt(np.sum(w * y * y) / sw)),
                'n': int(m.sum())}

    def rms(self, a):
        a = np.asarray(a, dtype='float64')
        m = (self.w > 0) & np.isfinite(a)
        if not m.any():
            return np.nan
        return float(np.sqrt(np.sum(self.w[m] * a[m] ** 2) / np.sum(self.w[m])))


def _advect(u, v, thk_dam, dx, dy):
    """-V.grad(h) as layer-mean temperature tendency in K/day, for any wind.

    The comparison field: the same quantity fb_boxology derives from the two
    contour families, computed the long way round from a wind the model
    actually carries, so the two can be held against each other in one unit.
    """
    h = np.asarray(thk_dam, dtype='float64') * 10.0
    dhdx = np.gradient(h, axis=1) / dx
    dhdy = np.gradient(h, axis=0) / dy
    return -(np.asarray(u) * dhdx + np.asarray(v) * dhdy) * 86400.0 \
        * bf.K_PER_DAM / 10.0


def fidelity(extent, score, tag, cycle=None, omega=False):
    """1. Does the two-contour shortcut measure the advection it claims?"""
    box = _fetch(extent, WIND_FIELDS + (OMEGA_FIELDS if omega else []),
                 tag + '_w', fhr=0, cycle=cycle)
    if box is None:
        raise SystemExit("boxology_verify: GFS winds unavailable; try again shortly.")
    lon2d, lat2d = _mesh(box)
    thk, mslp = _thk(box), _mslp(box)
    elev_m = bx.sample_dem(lon2d, lat2d)
    fc = bf.advance(lon2d, lat2d, mslp, thk, 1.0,
                    mslp_sigma=MSLP_SIGMA, elev_m=elev_m)
    inside = _window_mask(lon2d, lat2d, score)
    st = _Stats(bf.cos_weights(lat2d), (fc['status'] == bf.STATUS_OK) & inside)

    A = np.ma.filled(bx.advection_field(lon2d, lat2d, mslp, thk,
                                        mslp_sigma=MSLP_SIGMA, elev_m=elev_m),
                     np.nan)
    dx, dy = bx._metric(lon2d, lat2d)
    um = np.mean([box[f'u{lev}'] for lev in LEVELS], axis=0)
    vm = np.mean([box[f'v{lev}'] for lev in LEVELS], axis=0)
    ug, vg = bf.geostrophic_flow(lon2d, lat2d, mslp, MSLP_SIGMA)

    out = {'coverage': float(np.mean((fc['status'] == bf.STATUS_OK)[inside])),
           'against': {}, 'wind_speed_rms': {}}
    out['against']['model layer-mean (1000-500) advection'] = st.pair(
        A, _advect(um, vm, thk, dx, dy))
    for lev in LEVELS:
        out['against'][f'model {lev} mb advection'] = st.pair(
            A, _advect(box[f'u{lev}'], box[f'v{lev}'], thk, dx, dy))
    out['wind_speed_rms']['boxology V0 (geostrophic 1000)'] = st.rms(
        np.hypot(np.ma.filled(ug, np.nan), np.ma.filled(vg, np.nan)))
    out['wind_speed_rms']['model layer-mean (1000-500)'] = st.rms(np.hypot(um, vm))
    for lev in LEVELS:
        out['wind_speed_rms'][f'model {lev} mb'] = st.rms(
            np.hypot(box[f'u{lev}'], box[f'v{lev}']))
    out['boxology_advection_rms_kday'] = st.rms(A)
    if omega:
        # Warm advection against ASCENT (-omega), pointwise, and the Laplacian
        # form the QG omega equation would suggest. Both are here to be read,
        # not to be flattering.
        lap = (np.gradient(np.gradient(A, axis=1) / dx, axis=1) / dx
               + np.gradient(np.gradient(A, axis=0) / dy, axis=0) / dy)
        out['ascent'] = {}
        for lev in OMEGA_LEVELS:
            w = -np.asarray(box[f'w{lev}'], dtype='float64')
            out['ascent'][f'{lev} mb'] = {
                'A_vs_minus_omega': st.pair(A, w)['corr'],
                'minus_lapA_vs_minus_omega': st.pair(-lap, w)['corr']}
    return out


def attribution(extent, score, tag, scales=(1.5, 3.0, 6.0, 12.0), cycle=None):
    """2. Is that advection the local tendency? (centered difference)"""
    box = _fetch(extent, WIND_FIELDS, tag + '_w', fhr=TEND_HALF_H, cycle=cycle)
    before = _grab(extent, 0, tag, cycle)
    after = _grab(extent, 2 * TEND_HALF_H, tag, cycle)
    if box is None:
        raise SystemExit("boxology_verify: GFS winds unavailable; try again shortly.")
    lon2d, lat2d = _mesh(box)
    mslp = _mslp(box)
    elev_m = bx.sample_dem(lon2d, lat2d)
    fc = bf.advance(lon2d, lat2d, mslp, _thk(box), 1.0,
                    mslp_sigma=MSLP_SIGMA, elev_m=elev_m)
    inside = _window_mask(lon2d, lat2d, score)
    st = _Stats(bf.cos_weights(lat2d), (fc['status'] == bf.STATUS_OK) & inside)
    dx, dy = bx._metric(lon2d, lat2d)
    um = np.mean([box[f'u{lev}'] for lev in LEVELS], axis=0)
    vm = np.mean([box[f'v{lev}'] for lev in LEVELS], axis=0)

    rows = []
    for s in scales:
        thk = _thk(box, s)
        # K/day, centered on the analysis hour.
        tend = ((_thk(after, s) - _thk(before, s)) / (2.0 * TEND_HALF_H)
                * bf.K_PER_DAM * 24.0)
        A = np.ma.filled(bx.advection_field(lon2d, lat2d, mslp, thk,
                                            mslp_sigma=MSLP_SIGMA,
                                            elev_m=elev_m), np.nan)
        rows.append({'sigma': float(s), 'scale_km': float(s * 0.25 * 111.0),
                     'boxology_vs_tendency': st.pair(A, tend),
                     'true_advection_vs_tendency': st.pair(
                         _advect(um, vm, thk, dx, dy), tend),
                     'boxology_vs_true_advection': st.pair(
                         A, _advect(um, vm, thk, dx, dy))})
    return {'rows': rows, 'half_window_h': TEND_HALF_H}


def forecast_skill(extent, score, tag, leads=(6, 12, 24), cycle=None):
    """3. What the kinematic forecast is worth against persistence."""
    a = _grab(extent, 0, tag, cycle)
    lon2d, lat2d = _mesh(a)
    thk, mslp = _thk(a), _mslp(a)
    elev_m = bx.sample_dem(lon2d, lat2d)
    inside = _window_mask(lon2d, lat2d, score)
    w = bf.cos_weights(lat2d) * inside
    rows = []
    for h in leads:
        # A pinned (archive) cycle is scored against the VERIFYING ANALYSIS;
        # the live road can only reach its own run's forecast hour.
        if cycle is not None and int(h) % 6 == 0:
            truth = _thk(_grab(extent, 0, tag, cycle + timedelta(hours=int(h))))
        else:
            truth = _thk(_grab(extent, int(h), tag, cycle))
        fc = bf.advance(lon2d, lat2d, mslp, thk, h,
                        mslp_sigma=MSLP_SIGMA, elev_m=elev_m)
        raw = bf.skill(fc['thk'], truth, thk, weights=w)
        # The same operator scaled by the compensation measured in step 2 --
        # a MEASUREMENT of what calibration would buy, not a calibration. It
        # is not applied in fb_boxfcst and must not be.
        st = _Stats(bf.cos_weights(lat2d), inside & np.isfinite(fc['dthk']))
        slope = st.pair(fc['dthk'], truth - thk)['slope']
        scaled = bf.skill(thk + slope * fc['dthk'], truth, thk, weights=w)
        rows.append({'lead_h': float(h), 'skill': raw,
                     'fitted_slope': slope, 'skill_if_scaled': scaled,
                     'coverage': float(np.mean((fc['status'] == bf.STATUS_OK)[inside])),
                     'status': bf.status_tally(fc),
                     'boxes': bf.box_count_summary(fc)})
    return {'rows': rows, 'cycle': a['cycle'].strftime('%Y-%m-%d %HZ')}


def report(res):
    """The report, printed the way the findings actually rank."""
    log.info("BOXOLOGY ON REAL DATA — GFS %s, %s window",
             res.get('cycle', '?'), res['window'])
    f = res.get('fidelity')
    if f:
        log.info("")
        log.info("1. FIDELITY — does reading it off two contour families measure "
                 "the advection?")
        log.info("   boxology advection rms %.2f K/day, on %.0f%% of the window",
                 f['boxology_advection_rms_kday'], 100.0 * f['coverage'])
        for nm, p in f['against'].items():
            log.info("     vs %-38s corr %+0.3f   slope %+0.3f   rms %.2f",
                     nm, p['corr'], p['slope'], p['rms_y'])
        log.info("   advecting wind, rms m/s:")
        for nm, v in f['wind_speed_rms'].items():
            log.info("     %-40s %5.2f", nm, v)
    a = res.get('attribution')
    if a:
        log.info("")
        log.info("2. ATTRIBUTION — is that advection the local tendency? "
                 "(centered %dh)", 2 * a['half_window_h'])
        log.info("   %-12s %-22s %-22s %s", "smoothing",
                 "boxology vs tendency", "true adv vs tendency",
                 "boxology vs true adv")
        for r in a['rows']:
            log.info("   %4.0f km     corr %+0.3f slope %+0.3f   "
                     "corr %+0.3f slope %+0.3f   corr %+0.3f slope %+0.3f",
                     r['scale_km'],
                     r['boxology_vs_tendency']['corr'],
                     r['boxology_vs_tendency']['slope'],
                     r['true_advection_vs_tendency']['corr'],
                     r['true_advection_vs_tendency']['slope'],
                     r['boxology_vs_true_advection']['corr'],
                     r['boxology_vs_true_advection']['slope'])
        sl = a['rows'][0]['boxology_vs_tendency']['slope']
        if np.isfinite(sl) and sl > 0:
            log.info("   => advection over-predicts the tendency %.1fx; "
                     "about %.0f%% of it is canceled", 1.0 / sl, 100.0 * (1.0 - sl))
    s = res.get('skill')
    if s:
        log.info("")
        log.info("3. CONSEQUENCE — the kinematic forecast against persistence")
        for r in s['rows']:
            log.info("   +%3.0f h   RMSE %5.2f dam   persistence %5.2f   "
                     "SS %+0.3f   (scaled by %.2f: SS %+0.3f)   cover %3.0f%%",
                     r['lead_h'], r['skill']['rmse'],
                     r['skill']['rmse_persistence'], r['skill']['ss'],
                     r['fitted_slope'], r['skill_if_scaled']['ss'],
                     100.0 * r['coverage'])
        log.info("   The scaling is a MEASUREMENT, not a correction: it is "
                 "deliberately not applied in fb_boxfcst.")


def consistency(extent, draw, tag, cycle=None):
    """4. The three checks the write-up promises (preprint sec. 10), on one
    chart. `draw` is the frame the chart shows; `extent` the wider fetch.

      a. THERMAL-WIND CANCELLATION. -V.grad(h) with the whole layer-mean
         geostrophic wind V0 + VT/2 against V0 alone. Two ways: with VT taken
         from the SAME h that is being advected (the identity; the residual is
         round-off, because the discrete Jacobian J(h, h) vanishes term by
         term), and with VT taken from the two height surfaces each smoothed
         the way the chart smooths them (MSLP at 3.0, heights at 1.5), which is
         the honest measure of how far the chart's two plotted fields are from
         one mutually consistent atmosphere.
      b. INTERVAL INVARIANCE. The band a place is given should not depend on
         the contour intervals, in the limit of small boxes.
      c. BOX RATE AGAINST THE POINTWISE FIELD. Quantum over area against the
         box-mean of the pointwise Jacobian.
    """
    from scipy.ndimage import mean as _lmean, sum as _lsum
    box = _fetch(extent, Z_FIELDS, tag, fhr=0, cycle=cycle)
    if box is None:
        raise SystemExit("boxology_verify: GFS heights unavailable.")
    lon2d, lat2d = _mesh(box)
    thk, mslp = _thk(box), _mslp(box)
    elev_m = bx.sample_dem(lon2d, lat2d)
    inside = _window_mask(lon2d, lat2d, draw)
    w = bf.cos_weights(lat2d)
    out = {'draw_extent': list(draw)}

    # a. -------------------------------------------------------------------
    adv, taper, belt, _parts = bx._advection_parts(
        lon2d, lat2d, mslp, thk, mslp_sigma=MSLP_SIGMA, elev_m=elev_m)
    A0 = np.ma.filled(adv, np.nan)
    ok = inside & np.isfinite(A0) & (taper > 0)
    st = _Stats(w, ok)
    dx, dy = bx._metric(lon2d, lat2d)
    f = 2.0 * bx.OMEGA * np.sin(np.radians(lat2d))
    f = np.where(belt, np.nan, f)
    h = thk * 10.0
    hx, hy = np.gradient(h, axis=1) / dx, np.gradient(h, axis=0) / dy

    def _extra(hT):
        """Advection of h by HALF the thermal wind of thickness field hT, K/day,
        under the same taper as the box field."""
        tx, ty = np.gradient(hT, axis=1) / dx, np.gradient(hT, axis=0) / dy
        ut, vt = -(bx.G / f) * ty, (bx.G / f) * tx
        return -0.5 * (ut * hx + vt * hy) * 86400.0 * bf.K_PER_DAM / 10.0 * taper

    rms_A = st.rms(A0)
    same = _extra(h)
    z0_chart = (gaussian_filter(mslp, MSLP_SIGMA) - 1000.0) * bx.HPA_TO_GPM
    z500_chart = gaussian_filter(np.asarray(box['z500'], dtype='float64'),
                                 THK_SIGMA)
    mixed = _extra(z500_chart - z0_chart)
    out['thermal_wind'] = {
        'field_rms_kday': rms_A,
        'same_field': {'rms_frac': st.rms(same) / rms_A,
                       'max_frac': float(np.nanmax(np.abs(same[ok]))) / rms_A},
        'chart_smoothing': {'rms_frac': st.rms(mixed) / rms_A,
                            'max_frac': float(np.nanmax(np.abs(mixed[ok]))) / rms_A,
                            'corr_with_and_without':
                                st.pair(A0, A0 + mixed)['corr']}}

    # b. -------------------------------------------------------------------
    kw = dict(mslp_sigma=MSLP_SIGMA, elev_m=elev_m, whole_boxes=True)
    base = np.abs(bx.box_shading(lon2d, lat2d, mslp, thk, **kw)).astype(int)
    sgn0 = np.sign(bx.box_shading(lon2d, lat2d, mslp, thk, **kw))
    rows = []
    for dp, dh in ((2.0, 3.0), (2.0, 6.0), (4.0, 3.0), (8.0, 6.0),
                   (4.0, 12.0), (8.0, 12.0)):
        b = bx.box_shading(lon2d, lat2d, mslp, thk, dp, dh, **kw)
        other = np.abs(b).astype(int)
        m = inside & ((base > 0) | (other > 0))
        ww = w * m
        d = np.abs(other - base)
        both = m & (base > 0) & (other > 0)
        rows.append({'dp_hpa': dp, 'dthk_dam': dh,
                     'shaded_frac': float(np.sum(w * inside * (other > 0))
                                          / np.sum(w * inside)),
                     'moved_1plus': float(np.sum(ww * (d >= 1)) / np.sum(ww)),
                     'moved_2plus': float(np.sum(ww * (d >= 2)) / np.sum(ww)),
                     'sign_flips': float(np.sum(w * both * (np.sign(b) != sgn0))
                                         / max(np.sum(w * both), 1e-30))})
    out['interval_invariance'] = {
        'baseline': {'dp_hpa': bx.DEF_DP_HPA, 'dthk_dam': bx.DEF_DTHK_DAM,
                     'shaded_frac': float(np.sum(w * inside * (base > 0))
                                          / np.sum(w * inside))},
        'rows': rows}

    # c. -------------------------------------------------------------------
    labels, n = bx.label_boxes(lon2d, lat2d, mslp, thk, mslp_sigma=MSLP_SIGMA)
    idx = np.arange(1, n + 1)
    cell = np.abs(dx * dy)
    area = np.asarray(_lsum(np.nan_to_num(cell), labels, idx))
    fbar = np.asarray(_lmean(np.abs(2.0 * bx.OMEGA * np.sin(np.radians(lat2d))),
                             labels, idx))
    tbar = np.asarray(_lmean(np.nan_to_num(taper), labels, idx))
    mean_A = np.asarray(_lmean(np.nan_to_num(A0), labels, idx))
    cells = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    quantum = (bx._K * (bx.DEF_DP_HPA * bx.HPA_TO_GPM)
               * (bx.DEF_DTHK_DAM * 10.0) * 86400.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = quantum * tbar / (fbar * area)
    edge = set(np.unique(np.concatenate([labels[0, :], labels[-1, :],
                                         labels[:, 0], labels[:, -1]])))
    in_draw = np.asarray(_lmean(inside.astype('float64'), labels, idx)) > 0.999
    keep = (in_draw & np.isfinite(rate) & (rate >= bx.BANDS[0][0])
            & (tbar >= bx.FULL_TAPER)
            & ~np.isin(idx, list(edge)))
    whole = bx._whole_box_mask(labels, n, mslp, thk, bx.DEF_DP_HPA,
                               bx.DEF_DTHK_DAM, MSLP_SIGMA)
    out['regions'] = {'labelled': int(n), 'whole_boxes': int(whole.sum()),
                      'scored': int(keep.sum()),
                      'scored_whole': int((keep & whole).sum()),
                      'partial_share_of_scored_area': float(
                          area[keep & ~whole].sum() / max(area[keep].sum(), 1.0))}
    out['box_vs_pointwise'] = []
    for kind, lo, hi in (('whole', 4, 16), ('whole', 16, 36),
                         ('whole', 36, 10 ** 9), ('whole', 4, 10 ** 9),
                         ('partial', 4, 10 ** 9)):
        k = (keep & (cells >= lo) & (cells < hi)
             & (whole if kind == 'whole' else ~whole))
        if k.sum() < 3:
            continue
        ratio = rate[k] / np.abs(mean_A[k])
        lr = np.log(ratio[np.isfinite(ratio) & (ratio > 0)])
        out['box_vs_pointwise'].append({
            'kind': kind, 'cells': [int(lo), int(hi)], 'boxes': int(k.sum()),
            'median_ratio': float(np.exp(np.median(lr))),
            'p10_ratio': float(np.exp(np.percentile(lr, 10))),
            'p90_ratio': float(np.exp(np.percentile(lr, 90))),
            'log_corr': float(np.corrcoef(np.log(rate[k]),
                                          np.log(np.abs(mean_A[k])))[0, 1]),
            'same_band': float(np.mean(
                np.searchsorted([b[0] for b in bx.BANDS], rate[k], 'right')
                == np.searchsorted([b[0] for b in bx.BANDS],
                                   np.abs(mean_A[k]), 'right')))})
    return out


def run(window='conus', leads=(6, 12, 24), do_fidelity=True,
        do_attribution=True, do_skill=True, cycle=None, omega=False,
        quiet=False):
    win = WINDOWS[window]
    ext, score = win['fetch'], win['score']
    # fb_gfsbox keys its cache on (tag, cycle, fhr) and deliberately not on the
    # extent, so the tag has to carry the window or the Europe run would be
    # handed the Pacific's grid and score it without complaint.
    tag = f"boxfcst_{window}"
    res = {'window': window, 'score_extent': list(score)}
    if cycle is not None:
        res['cycle'] = cycle.strftime('%Y-%m-%d %HZ')
    if do_fidelity:
        res['fidelity'] = fidelity(ext, score, tag, cycle, omega)
    if do_attribution:
        res['attribution'] = attribution(ext, score, tag, cycle=cycle)
    if do_skill:
        res['skill'] = forecast_skill(ext, score, tag, leads, cycle)
        res['cycle'] = res['skill']['cycle']
    if not quiet:
        report(res)
    return res


def _spread(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)])
    if not v.size:
        return None
    return {'mean': float(v.mean()), 'min': float(v.min()),
            'max': float(v.max()), 'n': int(v.size)}


def summarize(runs):
    """Every (window, cycle) run reduced to mean and range per number -- the
    table the write-up quotes. One run is one weather pattern; the range is
    the honest error bar on 'one day's magnitudes'."""
    out = {'n_runs': len(runs), 'fidelity': {}, 'ascent': {},
           'attribution': {}, 'skill': {}}
    for nm in runs[0].get('fidelity', {}).get('against', {}):
        out['fidelity'][nm] = {
            k: _spread([r['fidelity']['against'][nm][k] for r in runs])
            for k in ('corr', 'slope')}
    for lev in runs[0].get('fidelity', {}).get('ascent', {}):
        out['ascent'][lev] = {
            k: _spread([r['fidelity']['ascent'][lev][k] for r in runs])
            for k in ('A_vs_minus_omega', 'minus_lapA_vs_minus_omega')}
    if runs[0].get('attribution'):
        for i, row in enumerate(runs[0]['attribution']['rows']):
            d = {}
            for key in ('boxology_vs_tendency', 'true_advection_vs_tendency'):
                for k in ('corr', 'slope'):
                    d[f'{key}.{k}'] = _spread(
                        [r['attribution']['rows'][i][key][k] for r in runs])
            d['geostrophic_tracks_better'] = float(np.mean(
                [r['attribution']['rows'][i]['boxology_vs_tendency']['corr']
                 > r['attribution']['rows'][i]['true_advection_vs_tendency']['corr']
                 for r in runs]))
            out['attribution'][f"{row['scale_km']:.0f} km"] = d
    if runs[0].get('skill'):
        for i, row in enumerate(runs[0]['skill']['rows']):
            out['skill'][f"+{row['lead_h']:.0f} h"] = {
                'ss': _spread([r['skill']['rows'][i]['skill']['ss'] for r in runs]),
                'beats_persistence': float(np.mean(
                    [r['skill']['rows'][i]['skill']['ss'] > 0 for r in runs]))}
    return out


def _cycle(text):
    return datetime.strptime(text, '%Y%m%d%H').replace(tzinfo=timezone.utc)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--window', default='conus',
                   help="one of %s, or a comma list" % ', '.join(sorted(WINDOWS)))
    p.add_argument('--leads', default='6,12,24')
    p.add_argument('--cycle', default=None,
                   help="pin cycles, YYYYMMDDHH[,YYYYMMDDHH...]; read from "
                        "NOAA's open-data archive instead of NOMADS")
    p.add_argument('--omega', action='store_true',
                   help="also correlate the box field with model ascent")
    p.add_argument('--consistency', default=None, metavar='W,E,S,N',
                   help="run the three consistency checks on this frame "
                        "(needs one --cycle) and stop")
    p.add_argument('--skip-fidelity', action='store_true')
    p.add_argument('--skip-attribution', action='store_true')
    p.add_argument('--skip-skill', action='store_true')
    p.add_argument('--json', help="write the full result dict here")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    cycles = [_cycle(c) for c in a.cycle.split(',')] if a.cycle else [None]

    if a.consistency:
        draw = tuple(float(v) for v in a.consistency.split(','))
        pad = 8.0
        ext = (max(-180.0, draw[0] - pad), min(180.0, draw[1] + pad),
               max(-85.0, draw[2] - pad), min(85.0, draw[3] + pad))
        res = consistency(ext, draw, 'boxchk', cycles[0])
        log.info(json.dumps(res, indent=1))
    else:
        leads = tuple(int(v) for v in a.leads.split(',') if v.strip())
        runs = []
        for wname in a.window.split(','):
            for cyc in cycles:
                runs.append(run(wname, leads, not a.skip_fidelity,
                                not a.skip_attribution, not a.skip_skill,
                                cycle=cyc, omega=a.omega))
        res = runs[0] if len(runs) == 1 else {'runs': runs,
                                              'summary': summarize(runs)}
        if len(runs) > 1:
            log.info("")
            log.info("SUMMARY over %d runs", len(runs))
            log.info(json.dumps(res['summary'], indent=1))
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump(res, fh, indent=1, default=str)
        log.info("wrote %s", a.json)
    return res


if __name__ == "__main__":
    main()
