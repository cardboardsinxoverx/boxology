#!/usr/bin/env python3
"""boxology_figures.py — the preprint's chart figures, from the archive.

A standalone chart: MSLP isobars, 1000-500 hPa thickness lines, and the box
shading of fb_boxology over them, from one archived GFS 0.25 deg analysis
(fb_gfsarchive; NOAA's open-data bucket, any cycle since 2021). Nothing else of
the suite is involved -- no blend, no observations, no GUI -- so a figure made
here can be made again by anyone with the three files and an internet
connection, which is the property a paper's figure needs and a screenshot of
the app does not have.

The fields are the chart's own recipe: MSLP = 1000 + Z1000/8 (smoothed 3.0 by
the shading, and drawn at that smoothing), thickness (Z500 - Z1000) smoothed
1.5. Every premise taper is on: latitude, curvature, and terrain off ETOPO1's
ice surface.

    python boxology_figures.py paper --out ~/Downloads/boxology_v3/figs
    python boxology_figures.py chart --cycle 2026090712 --extent=-140,-85,12,52 \
        --out fig.png
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.ndimage import gaussian_filter

import fb_boxfcst as bf
import fb_boxology as bx
import fb_gfsarchive as ga
import fb_gfsbox as gb

log = logging.getLogger("boxology_figures")

Z_FIELDS = [gb.isobaric('z500', 'HGT', 'gh', 500),
            gb.isobaric('z1000', 'HGT', 'gh', 1000)]
THK_SIGMA, MSLP_SIGMA = 1.5, 3.0
DP_HPA = 2.0              # the globe's isobar interval; the RATE does not depend on it
PAD = 8.0                 # fetch past the frame so no drawn box is a cut box
# With trails the data box must reach far upwind of the frame, or every parcel
# near the upwind edge is withheld as "came from off the chart" (12 h at
# 30 m/s is 1,300 km). Degrees of longitude, latitude.
TRAIL_MARGIN = (30.0, 14.0)
TRAIL_HOURS = 12.0
DTHK_DAM = 6.0

INK = '#1b2430'
THK_COLOR = '#8a4b12'
THK_540 = '#1f4fbf'
LAND, SEA = '#f1efe9', '#ffffff'


def fields(cycle, draw, fhr=0, margin=(PAD, PAD)):
    ext = (max(-180.0, draw[0] - margin[0]), min(180.0, draw[1] + margin[0]),
           max(-85.0, draw[2] - margin[1]), min(85.0, draw[3] + margin[1]))
    box = ga.fetch(ext, Z_FIELDS, fhr=fhr, cycle=cycle)
    if box is None:
        raise SystemExit(f"no archive GFS for {cycle:%Y%m%d%H} f{fhr:03d}")
    lon2d, lat2d = np.meshgrid(box['lon'], box['lat'])
    z1000 = np.asarray(box['z1000'], dtype='float64')
    thk = gaussian_filter((np.asarray(box['z500'], dtype='float64') - z1000)
                          / 10.0, THK_SIGMA)
    mslp = 1000.0 + z1000 / 8.0
    return lon2d, lat2d, mslp, thk


def draw_panel(ax, cycle, draw, fhr=0, marks=(), labels=True, tapers=True,
               title=None, dp=DP_HPA, track=None, trails=True, density=24.0,
               jet=None):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.patheffects as pe

    pc = ccrs.PlateCarree()
    lon2d, lat2d, mslp, thk = fields(
        cycle, draw, fhr, TRAIL_MARGIN if trails else (PAD, PAD))
    elev = bx.sample_dem(lon2d, lat2d) if tapers else None

    ax.set_extent(draw, crs=pc)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor=SEA, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=LAND, zorder=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#6b7480',
                   linewidth=0.45, zorder=2.0)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='#a3aab3',
                   linewidth=0.3, zorder=2.0)

    band = bx.shade_boxes(ax, lon2d, lat2d, mslp, thk, transform=pc,
                          mslp_sigma=MSLP_SIGMA, elev_m=elev,
                          curvature_taper=tapers, dp_hpa=dp, whole_boxes=True)

    p_s = gaussian_filter(mslp, MSLP_SIGMA)
    lo, hi = np.floor(p_s.min() / dp) * dp, np.ceil(p_s.max() / dp) * dp
    cs = ax.contour(lon2d, lat2d, p_s, levels=np.arange(lo, hi + 1, dp),
                    colors=INK, linewidths=0.55, zorder=2.3, transform=pc)
    tl = np.arange(np.floor(thk.min() / 6) * 6, thk.max() + 6, 6.0)
    ct = ax.contour(lon2d, lat2d, thk, levels=[v for v in tl if v != 540],
                    colors=THK_COLOR, linewidths=0.7, linestyles='--',
                    zorder=2.2, transform=pc)
    c5 = ax.contour(lon2d, lat2d, thk, levels=[540.0], colors=THK_540,
                    linewidths=1.1, linestyles='--', zorder=2.25, transform=pc)
    if labels:
        halo = [pe.withStroke(linewidth=1.6, foreground='white')]
        for c, col, fs in ((cs, INK, 5.5), (ct, THK_COLOR, 5.5), (c5, THK_540, 6)):
            for t in ax.clabel(c, levels=[v for v in c.levels if v % 4 == 0 or c is not cs], fmt='%d', fontsize=fs, inline=True,
                               inline_spacing=2, colors=col):
                t.set_path_effects(halo)

    if trails:
        _draw_trails(ax, lon2d, lat2d, mslp, thk, elev if tapers else None,
                     draw, pc, density=density)

    if track:
        # The agency track: the whole line thin, a ring on every 00Z fix with
        # its day of the month, and the LAST fix crossed -- after it the
        # agency was no longer following the system, and the chart says so.
        halo = [pe.withStroke(linewidth=2, foreground='white')]
        ax.plot([t[1] for t in track], [t[2] for t in track], '-', color='k',
                lw=1.0, zorder=4.8, transform=pc, path_effects=halo)
        said = []
        for t, lon, lat in track:
            if t.hour == 0 and draw[0] < lon < draw[1] and draw[2] < lat < draw[3]:
                ax.plot(lon, lat, 'o', mfc='white', mec='k', ms=3.6, mew=0.8,
                        zorder=4.9, transform=pc)
                # A looping track stacks its day numbers; say one only where
                # there is room for it.
                if any(np.hypot(lon - a, lat - b) < 2.2 for a, b in said):
                    continue
                said.append((lon, lat))
                ax.annotate(f'{t.day:02d}', (lon, lat), xytext=(-9, -8),
                            textcoords='offset points', fontsize=5.5,
                            zorder=5, path_effects=halo, transform=pc)
        ax.plot(track[-1][1], track[-1][2], 'x', color='k', ms=6, mew=1.6,
                zorder=5, transform=pc)
    for lon, lat, text in marks:
        ax.plot(lon, lat, marker='x', ms=6, mew=1.6, color='k', zorder=5,
                transform=pc)
        if text:
            ax.annotate(text, (lon, lat), xytext=(5, 5),
                        textcoords='offset points', fontsize=6.5, zorder=5,
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground='white')],
                        transform=pc)

    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color='#9aa3ad',
                      linestyle=':', zorder=1.0)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 6.5, 'color': '#444'}
    valid = cycle + timedelta(hours=fhr)
    # A text, not set_title: cartopy's gridliner re-places an axes title on
    # draw and on these frames puts it off the figure.
    ax.text(0.0, 1.015, title or f"GFS 0.25° analysis  {valid:%d %b %Y %H%MZ}",
            transform=ax.transAxes, fontsize=8.5, ha='left', va='bottom')
    return band


def _save(fig, out, dpi):
    """Save, then trim the white margin. (bbox_inches='tight' cannot be used:
    with cartopy axes it measures the legend alone and crops the maps away.)"""
    from PIL import Image, ImageChops
    fig.savefig(out, dpi=dpi, facecolor='white')
    im = Image.open(out).convert('RGB')
    box = ImageChops.difference(im, Image.new('RGB', im.size, 'white')).getbbox()
    if box:
        m = int(0.06 * dpi)
        im.crop((max(box[0] - m, 0), max(box[1] - m, 0),
                 min(box[2] + m, im.width), min(box[3] + m, im.height))).save(out)


def _trail_cmap():
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list(
        'boxtrail_paper', [bx.COLD_COLOR, '#9fb4d6', '#b9b9b9', '#e2a59f',
                           bx.WARM_COLOR])


JET_COLOR = '#2e8b57'
JET_LEVELS = (70.0, 90.0, 110.0, 130.0, 150.0)     # kt


def _draw_jet(ax, cycle, draw, fhr, level, pc):
    """Where the jet is: isotachs at `level` hPa, green, under everything.
    The surface chart reads better with the jet known -- the couplets sit under
    its entrance and exit regions -- and green is the one hue the box shading,
    the thickness lines and the trails leave free."""
    import matplotlib.colors as mcolors
    flds = [gb.isobaric(f'u{level}', 'UGRD', 'u', level),
            gb.isobaric(f'v{level}', 'VGRD', 'v', level)]
    ext = (max(-180.0, draw[0] - 2), min(180.0, draw[1] + 2),
           max(-85.0, draw[2] - 2), min(85.0, draw[3] + 2))
    box = ga.fetch(ext, flds, fhr=fhr, cycle=cycle)
    if box is None:
        log.warning("no %d hPa wind in the archive for this hour", level)
        return
    lon2d, lat2d = np.meshgrid(box['lon'], box['lat'])
    kt = gaussian_filter(np.hypot(box[f'u{level}'], box[f'v{level}']), 1.5) \
        * 1.94384
    fills = [mcolors.to_rgba(JET_COLOR, a) for a in (0.14, 0.23, 0.32, 0.41, 0.5)]
    ax.contourf(lon2d, lat2d, kt, levels=list(JET_LEVELS) + [400.0],
                colors=fills, zorder=1.0, transform=pc)
    ax.contour(lon2d, lat2d, kt, levels=JET_LEVELS, colors=JET_COLOR,
               linewidths=0.6, zorder=1.05, transform=pc)


def draw_jet_panel(ax, cycle, draw, level=300, fhr=0, title=None, track=None):
    """The upper-air companion: `level` hPa heights and isotachs on the same
    frame as the surface chart, as its OWN panel (Evan, 2026-09-18: on the
    surface chart it was "way too busy")."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.patheffects as pe
    pc = ccrs.PlateCarree()
    ax.set_extent(draw, crs=pc)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor=SEA, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=LAND, zorder=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#6b7480',
                   linewidth=0.45, zorder=2.0)
    _draw_jet(ax, cycle, draw, fhr, int(level), pc)
    ext = (max(-180.0, draw[0] - 2), min(180.0, draw[1] + 2),
           max(-85.0, draw[2] - 2), min(85.0, draw[3] + 2))
    box = ga.fetch(ext, [gb.isobaric(f'z{level}', 'HGT', 'gh', int(level))],
                   fhr=fhr, cycle=cycle)
    if box is not None:
        lon2d, lat2d = np.meshgrid(box['lon'], box['lat'])
        z = gaussian_filter(np.asarray(box[f'z{level}'], dtype='float64'),
                            1.5) / 10.0
        cs = ax.contour(lon2d, lat2d, z,
                        levels=np.arange(600, 1300, 12.0), colors=INK,
                        linewidths=0.7, zorder=2.3, transform=pc)
        for t in ax.clabel(cs, fmt='%d', fontsize=5.5, inline=True,
                           inline_spacing=2):
            t.set_path_effects([pe.withStroke(linewidth=1.6,
                                              foreground='white')])
    if track:
        ax.plot([t[1] for t in track], [t[2] for t in track], '-', color='k',
                lw=1.0, zorder=4.8, transform=pc,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        ax.plot(track[-1][1], track[-1][2], 'x', color='k', ms=6, mew=1.6,
                zorder=5, transform=pc)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color='#9aa3ad',
                      linestyle=':', zorder=1.0)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 6.5, 'color': '#444'}
    valid = cycle + timedelta(hours=fhr)
    ax.text(0.0, 1.015, title or f"{int(level)} hPa  {valid:%d %b %Y %H%MZ}",
            transform=ax.transAxes, fontsize=8.5, ha='left', va='bottom')


def _draw_trails(ax, lon2d, lat2d, mslp, thk, elev, draw, pc, span=2.5,
                 label_top=10, density=24.0):
    """boxology_origin's trails on the paper chart: 12-h back-trajectories down
    the isobar channels, colored and numbered by the boluses delivered, and
    hatching where the premise failed somewhere upstream."""
    import matplotlib
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as pe
    from matplotlib.collections import LineCollection

    fc = bf.advance(lon2d, lat2d, mslp, thk, TRAIL_HOURS,
                    mslp_sigma=MSLP_SIGMA, thk_sigma=0.0, elev_m=elev)
    held = (fc['status'] != bf.STATUS_OK).astype('float64')
    if held.any():
        matplotlib.rcParams['hatch.color'] = '#9aa3ad'
        matplotlib.rcParams['hatch.linewidth'] = 0.3
        ax.contourf(lon2d, lat2d, held, levels=[0.5, 1.5], colors='none',
                    hatches=['////'], zorder=1.25, transform=pc)
    inside = ((lon2d >= draw[0]) & (lon2d <= draw[1])
              & (lat2d >= draw[2]) & (lat2d <= draw[3]))
    cols = max(int(inside[lat2d.shape[0] // 2, :].sum()), 1)
    stride = int(np.clip(round(cols / float(density)), 4, 40))
    trails = bf.trajectories(lon2d, lat2d, mslp, thk, TRAIL_HOURS,
                             stride=stride, min_boxes=0.35,
                             mslp_sigma=MSLP_SIGMA, thk_sigma=0.0,
                             elev_m=elev, dthk_dam=DTHK_DAM)
    trails = [t for t in trails
              if draw[0] <= t['lon'][-1] <= draw[1]
              and draw[2] <= t['lat'][-1] <= draw[3]]
    if not trails:
        return
    cmap, norm = _trail_cmap(), mcolors.Normalize(-span, span)
    segs = [np.column_stack([t['lon'], t['lat']]) for t in trails]
    vals = np.array([t['boxes'] for t in trails])
    ax.add_collection(LineCollection(segs, colors='#1b2430', linewidths=2.3,
                                     zorder=2.55, capstyle='round',
                                     transform=pc))
    ax.add_collection(LineCollection(segs, array=vals, cmap=cmap, norm=norm,
                                     linewidths=1.35, zorder=2.6,
                                     capstyle='round', transform=pc))
    tr = pc._as_mpl_transform(ax)
    for t in trails:
        if t['lon'].size < 3:
            continue
        ax.annotate('', xy=(t['lon'][-1], t['lat'][-1]),
                    xytext=(t['lon'][-3], t['lat'][-3]), xycoords=tr,
                    textcoords=tr, zorder=2.7,
                    arrowprops=dict(arrowstyle='-|>', mutation_scale=7.5,
                                    color=cmap(norm(t['boxes'])), lw=0.6,
                                    ec='#1b2430'))
    floor = 0.06 * max(draw[1] - draw[0], draw[3] - draw[2])
    placed = []
    for t in sorted(trails, key=lambda t: -abs(t['boxes'])):
        if len(placed) >= label_top:
            break
        x, y = float(t['lon'][-1]), float(t['lat'][-1])
        if not (draw[0] + 0.04 * (draw[1] - draw[0]) < x
                < draw[1] - 0.04 * (draw[1] - draw[0])
                and y < draw[3] - 0.06 * (draw[3] - draw[2])):
            continue
        if any((x - a) ** 2 + (y - b) ** 2 < floor ** 2 for a, b in placed):
            continue
        placed.append((x, y))
        ax.text(x, y, f"{t['boxes']:+.1f}", fontsize=6.2, fontweight='bold',
                color='#10161f', ha='center', va='bottom', zorder=2.8,
                transform=pc,
                path_effects=[pe.withStroke(linewidth=1.9, foreground='white')])


def _key(fig, y=0.02, dp=DP_HPA, trails=True, jet=None):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    def chips(color, word):
        out = []
        for i in (0, 3, 7):
            lo_, hi_ = bx.BANDS[i][0], bx.BANDS[i][1]
            side = bx.box_side_for_rate(lo_, lat=45.0, dp_hpa=dp)
            rate = (f'>{lo_:.0f}' if not np.isfinite(hi_)
                    else f'{lo_:.0f}\u2013{hi_:.0f}')
            out.append(Patch(facecolor=color, alpha=bx.FILL_ALPHA[i],
                             edgecolor=color,
                             label=f'{word}, {rate} K/day '
                                   f'(box \u2272 {side:.0f} km at 45\u00b0)'))
        return out

    warm, cold = chips(bx.WARM_COLOR, 'warm'), chips(bx.COLD_COLOR, 'cold')
    lines = [Line2D([], [], color=INK, lw=0.9, label=f'MSLP, every {dp:g} hPa'),
             Line2D([], [], color=THK_COLOR, lw=0.9, ls='--',
                    label='1000\u2013500 hPa thickness, every 6 dam'),
             Line2D([], [], color=THK_540, lw=1.2, ls='--', label='540 dam')]
    if trails:
        warm.append(Line2D([], [], color=bx.WARM_COLOR, lw=1.6,
                           label='12-h trail, warmer air arriving'))
        cold.append(Line2D([], [], color=bx.COLD_COLOR, lw=1.6,
                           label='12-h trail, colder air arriving'))
        lines.append(Patch(facecolor='none', edgecolor='#9aa3ad',
                           hatch='////', label='no trail: premise fails along the path'))
    if jet:
        lines.append(Patch(facecolor=JET_COLOR, alpha=0.3, edgecolor=JET_COLOR,
                           label=f'{int(jet)} hPa wind \u2265 70 kt (every 20)'))
        while len(warm) < len(lines):
            warm.append(Line2D([], [], lw=0, label=' '))
            cold.append(Line2D([], [], lw=0, label=' '))
    fig.legend(handles=warm + cold + lines, loc='lower center', ncol=3,
               fontsize=6.0, frameon=False, bbox_to_anchor=(0.5, y + 0.018
                                                            if trails else y),
               columnspacing=1.2, handlelength=2.0)
    if trails:
        fig.text(0.5, y, 'Numbers: boluses the flow delivered in 12 h '
                         '(1 bolus = one thickness line crossed = 6 dam = '
                         '2.96 K of layer-mean temperature). Transport, not a '
                         'forecast.', ha='center', va='bottom', fontsize=6.0)


def chart(cycle, draw, out, fhr=0, marks=(), tapers=True, dpi=220, title=None,
          dp=DP_HPA, track=None, jet=None):
    """One surface chart; with `jet`, the upper-air panel under it as (b)."""
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    w = 7.4
    aspect = (draw[3] - draw[2]) / (draw[1] - draw[0])
    rows = 2 if jet else 1
    ph = w * 0.92 * aspect
    fh = rows * (ph + 0.3) + 1.3
    fig = plt.figure(figsize=(w, fh))
    proj = ccrs.PlateCarree(central_longitude=0.5 * (draw[0] + draw[1]))
    valid = cycle + timedelta(hours=fhr)
    for i in range(rows):
        y0 = (1.25 + (rows - 1 - i) * (ph + 0.3)) / fh
        ax = fig.add_axes([0.05, y0, 0.92, ph / fh], projection=proj)
        if i == 0:
            draw_panel(ax, cycle, draw, fhr, marks, tapers=tapers, dp=dp,
                       track=track,
                       title=title or (f"{'(a)  ' if jet else ''}GFS 0.25\u00b0 "
                                       f"analysis  {valid:%d %b %Y %H%MZ}"))
        else:
            draw_jet_panel(ax, cycle, draw, jet, fhr, track=track,
                           title=f"(b)  {int(jet)} hPa heights and wind  "
                                 f"{valid:%d %b %Y %H%MZ}")
    _key(fig, dp=dp, jet=jet, y=0.012)
    _save(fig, out, dpi)
    plt.close(fig)
    log.info("wrote %s", out)
    return out


def sequence(cycles, draw, out, marks=None, dpi=220, ncols=2, dp=DP_HPA,
             track=None, jet=None, jet_panels=False):
    """Panels of one frame through time -- the Krovanh figure."""
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(cycles)
    nrows = int(np.ceil(n / ncols))
    aspect = (draw[3] - draw[2]) / (draw[1] - draw[0])
    w = 7.4
    pw = w / ncols
    fig = plt.figure(figsize=(w, nrows * (pw * aspect + 0.12) + 1.25))
    proj = ccrs.PlateCarree(central_longitude=0.5 * (draw[0] + draw[1]))
    for i, cyc in enumerate(cycles):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection=proj)
        if jet_panels:
            draw_jet_panel(ax, cyc, draw, jet or 300,
                           track=[t for t in (track or []) if t[0] <= cyc],
                           title=f"({'abcdefgh'[i]})  {int(jet or 300)} hPa  "
                                 f"{cyc:%d %b %Y %H%MZ}")
            continue
        draw_panel(ax, cyc, draw, marks=(marks or {}).get(cyc, ()), dp=dp,
                   density=12.0,
                   track=[t for t in (track or []) if t[0] <= cyc],
                   title=f"({'abcdefgh'[i]})  {cyc:%d %b %Y %H%MZ}")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.965,
                        bottom=(0.3 if jet_panels else 1.0)
                        / fig.get_figheight() + 0.03,
                        wspace=0.10, hspace=0.10)
    if jet_panels:
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor=JET_COLOR, alpha=0.3,
                                  edgecolor=JET_COLOR,
                                  label=f'{int(jet or 300)} hPa wind, 70 kt and '
                                        f'up (every 20 kt); heights every '
                                        f'12 dam; line: JMA track to date')],
                   loc='lower center', fontsize=6.3, frameon=False,
                   bbox_to_anchor=(0.5, 0.02))
    else:
        _key(fig, y=0.005, dp=dp)
    _save(fig, out, dpi)
    plt.close(fig)
    log.info("wrote %s", out)
    return out


def compare(cycle, draw, out, dpi=220, marks=(), jet=None):
    """The same analysis with the premise tapers off (a) and on (b): what the
    validity envelope removes, and what survives it."""
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    w = 7.4
    aspect = (draw[3] - draw[2]) / (draw[1] - draw[0])
    fig = plt.figure(figsize=(w, 2 * (w * 0.92 * aspect + 0.3) + 1.2))
    proj = ccrs.PlateCarree(central_longitude=0.5 * (draw[0] + draw[1]))
    for i, (tap, word) in enumerate(((False, 'curvature and terrain tapers '
                                             'off, boxes only'),
                                     (True, 'latitude, curvature and terrain '
                                            'tapers on'))):
        ax = fig.add_subplot(2, 1, i + 1, projection=proj)
        draw_panel(ax, cycle, draw, tapers=tap, marks=marks, trails=tap,
                   title=f"({'ab'[i]})  {cycle:%d %b %Y %H%MZ} \u2014 {word}")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.97,
                        bottom=1.2 / fig.get_figheight() + 0.03, hspace=0.12)
    _key(fig, y=0.005)
    _save(fig, out, dpi)
    plt.close(fig)
    log.info("wrote %s", out)
    return out


def house_chart(cycle, draw, out, title, track=None, hours=12.0, dpi=200,
                margin=(30.0, 14.0)):
    """The suite's own trajectory chart (boxology_origin.render) on an ARCHIVED
    analysis: boxes, 12-h trails with their bolus counts, hatching where the
    premise failed upstream -- the house product, fed these fields instead of
    fetching the newest run. The data box is wider than the frame by `margin`
    degrees so the trails have somewhere to come from."""
    import shutil
    import tempfile
    import boxology_origin as bo

    ext = (draw[0] - margin[0], draw[1] + margin[0],
           max(-85.0, draw[2] - margin[1]), min(85.0, draw[3] + margin[1]))
    if ext[1] > 180.0 or ext[0] < -180.0:
        # fb_boxology.sample_dem and the trajectory code want signed degrees.
        ext = (max(-180.0, ext[0]), min(180.0, ext[1]), ext[2], ext[3])
    box = ga.fetch(ext, Z_FIELDS, cycle=cycle)
    if box is None:
        raise SystemExit(f"no archive GFS for {cycle:%Y%m%d%H}")
    lon2d, lat2d = np.meshgrid(box['lon'], box['lat'])
    z1000 = np.asarray(box['z1000'], dtype='float64')
    thk = gaussian_filter((np.asarray(box['z500'], dtype='float64') - z1000)
                          / 10.0, THK_SIGMA)
    mslp = 1000.0 + z1000 / 8.0
    tmp = tempfile.mkdtemp(prefix='boxfig_')
    try:
        path = bo.render(hours=hours, out_dir=tmp, dpi=dpi,
                         fields=(lon2d, lat2d, mslp, thk), draw=draw,
                         valid=cycle, title=title, file_tag='boxfig',
                         air_masses=False, track=track)
        shutil.move(path, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log.info("wrote %s", out)
    return out


def tile(paths, out, ncols=2, gap=24, bg=(10, 20, 29)):
    """Several house charts on one sheet, row-major."""
    from PIL import Image
    ims = [Image.open(p).convert('RGB') for p in paths]
    w, h = ims[0].size
    nrows = -(-len(ims) // ncols)
    sheet = Image.new('RGB', (ncols * w + (ncols - 1) * gap,
                              nrows * h + (nrows - 1) * gap), bg)
    for i, im in enumerate(ims):
        sheet.paste(im.resize((w, h)), ((i % ncols) * (w + gap),
                                        (i // ncols) * (h + gap)))
    sheet.save(out)
    log.info("wrote %s", out)
    return out


def read_track(path):
    """[(time, lon, lat)] from the track CSV (time_utc,lat,lon,...)."""
    out = []
    with open(os.path.expanduser(path)) as fh:
        for ln in fh:
            if ln.startswith('#') or ln.startswith('time_utc') or not ln.strip():
                continue
            c = ln.split(',')
            out.append((datetime.strptime(c[0], '%Y-%m-%dT%H:%MZ').replace(
                tzinfo=timezone.utc), float(c[2]), float(c[1])))
    return out


def _cyc(text):
    return datetime.strptime(text, '%Y%m%d%H').replace(tzinfo=timezone.utc)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('chart')
    c.add_argument('--cycle', required=True)
    c.add_argument('--extent', required=True, help='W,E,S,N')
    c.add_argument('--out', required=True)
    c.add_argument('--no-tapers', action='store_true')
    hs = sub.add_parser('house')
    hs.add_argument('--cycle', required=True)
    hs.add_argument('--extent', required=True)
    hs.add_argument('--out', required=True)
    hs.add_argument('--title', required=True)
    k = sub.add_parser('compare')
    k.add_argument('--cycle', required=True)
    k.add_argument('--extent', required=True)
    k.add_argument('--out', required=True)
    s = sub.add_parser('sequence')
    s.add_argument('--cycles', required=True)
    s.add_argument('--extent', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--jet-panels', action='store_true',
                   help='draw the upper-air panels instead of the surface ones')
    for q in (c, k, s, hs):
        q.add_argument('--dp', type=float, default=DP_HPA,
                       help='isobar interval, hPa (the rate does not depend on it)')
        q.add_argument('--track', default=None, help='track CSV to draw')
        q.add_argument('--jet', type=int, default=None, metavar='HPA',
                       help='also draw isotachs at this level (200 or 300)')
    a = p.parse_args(argv)
    track = read_track(a.track) if a.track else None
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    ext = tuple(float(v) for v in a.extent.split(','))
    if a.cmd == 'chart':
        chart(_cyc(a.cycle), ext, os.path.expanduser(a.out),
              tapers=not a.no_tapers, dp=a.dp, track=track, jet=a.jet)
    elif a.cmd == 'house':
        house_chart(_cyc(a.cycle), ext, os.path.expanduser(a.out), a.title,
                    track=track)
    elif a.cmd == 'compare':
        compare(_cyc(a.cycle), ext, os.path.expanduser(a.out), jet=a.jet)
    else:
        sequence([_cyc(v) for v in a.cycles.split(',')], ext,
                 os.path.expanduser(a.out), dp=a.dp, track=track, jet=a.jet,
                 jet_panels=a.jet_panels)


if __name__ == "__main__":
    main()
