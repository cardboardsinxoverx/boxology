# Boxology

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22844862.svg)](https://doi.org/10.5281/zenodo.22844862)

Code, data and results for the preprint **Boxology: the isobar-thickness box
count as an exact advection integral, and its rendering** (E. J. Lane, 2026;
revision 3, 18 September 2026). The manuscript is in [`paper/`](paper/).

Every release is archived on Zenodo;
<https://doi.org/10.5281/zenodo.22844862> always resolves to the newest one.

On a surface chart the isobars and the 1000-500 hPa thickness lines cut the map
into boxes. The boxes are small where thermal advection is strong, and the sense
of the crossing says whether it is warm or cold. This code labels and shades
those boxes on a 0.25-degree GFS analysis, draws 12-hour back-trajectories down
the isobar channels, and scores the result against the model's own tendencies.

This is the renderer used by the FrostByte weather visualization suite, released
here so the paper's figures and numbers can be reproduced without the suite.

## What is here

| Path | What |
|---|---|
| `fb_boxology.py` | The renderer: constants, box labeling, the whole-box rule, tapers, shading |
| `fb_boxfcst.py` | The trajectory operator behind the trails |
| `boxology_verify.py` | The verification and consistency harness (sections 9-10) |
| `boxology_figures.py` | The script that made the paper's chart figures |
| `fb_gfsarchive.py` | Reads GFS analyses by byte range from the NOAA archive on AWS |
| `fb_gfsbox.py` | The field table and answer shape the archive reader shares with the suite |
| `paper/` | The manuscript (`.pdf`, `.tex`) and its six figures |
| `results/` | The per-run JSON and log behind every number in sections 9-10 |
| `data/krovanh_jma_track.csv` | Typhoon Krovanh (2624), JMA track data as archived by Digital Typhoon, National Institute of Informatics (https://agora.ex.nii.ac.jp/digital-typhoon/), retrieved 18 September 2026 |

## Run it

Python 3.10 or later. The box labeling and shading need NumPy, SciPy, Matplotlib
and Cartopy; the harness and the figure script add `requests` and `cfgrib`
(which needs ecCodes).

    pip install -r requirements.txt

    T=data/krovanh_jma_track.csv ; F=out ; mkdir -p $F
    python boxology_figures.py compare  --cycle 2026090712 --extent=-150,-85,10,50 --out $F/fig1_conus_sep07.png
    python boxology_figures.py chart    --cycle 2026090712 --extent=90,172,5,50 --track $T --jet 300 --out $F/fig2_wpac_sep07.png
    python boxology_figures.py chart    --cycle 2026090712 --extent=10,178,-65,-10 --dp 4 --jet 300 --out $F/fig3_sh_sep07.png
    python boxology_figures.py sequence --cycles 2026090700,2026090712,2026090800,2026090900 --extent=122,158,20,46 --track $T --out $F/fig4_krovanh_sequence.png
    python boxology_figures.py sequence --cycles 2026090700,2026090712,2026090800,2026090900 --extent=122,158,20,46 --track $T --jet 300 --jet-panels --out $F/fig4b_krovanh_300hpa.png
    python boxology_figures.py chart    --cycle 2026091800 --extent=-126,-66,23,52 --out $F/fig6_transport_conus.png

    # sec. 9: 12 cycles x 4 windows (about 30 min, ~300 MB of byte-range reads, cached in cache/)
    python boxology_verify.py --window conus,europe,sindian,satlantic --omega \
        --cycle 2025101500,2025111512,2025121500,2026011512,2026021500,2026031512,2026041500,2026051512,2026061500,2026071512,2026081500,2026091512 \
        --json verify_12cycles_4windows.json
    # sec. 10: the three consistency checks on the Figure 3 case
    python boxology_verify.py --cycle 2026090712 --consistency=10,172,-65,-10 --json consistency_fig3_2026090712.json

Any past date works the same way (the archive goes back to 2021):
`boxology_figures.py chart --cycle YYYYMMDDHH --extent=W,E,S,N --out x.png`
(add `--jet 300` for the upper-air panel). Downloads are cached in `cache/`
beside the code; set `FROSTBYTE_CACHE` to put them elsewhere.

## One difference from the paper's figures

**The terrain taper.** In the suite, `fb_boxology.sample_dem` reads ETOPO1 (ice
surface) through the suite's elevation reader, which is not part of this
release. Without it the code says so and skips the terrain taper; the latitude
and curvature tapers are unaffected, and so is every number in sections 9-10.
To apply it, pass any elevation array on the chart's grid as `elev_m`.

`boxology_figures.py house` draws the suite's own transport product and needs
the rest of FrostByte; no figure in the paper uses it.

## Data

All fields are operational NCEP GFS 0.25-degree analyses, read from the NOAA Open
Data Dissemination archive on Amazon Web Services
(https://noaa-gfs-bdp-pds.s3.amazonaws.com). Elevation in the paper is ETOPO1,
ice surface (NOAA National Centers for Environmental Information).

## License and citation

Copyright (C) 2026 Evan James Lane. The code is under the GNU General Public
License, version 3 or later (`LICENSE`); the manuscript and figures in `paper/`
are under CC BY 4.0 (`paper/LICENSE`), the same license the preprint carries on
EarthArXiv. `NOTICE.txt` says why the code is GPL and which third-party data
carries its own terms.

Lane, E. J., 2026: *Boxology: the isobar-thickness box count as an exact
advection integral, and its rendering.* Preprint, revision 3.
<https://doi.org/10.5281/zenodo.22844862>

The companion technical reference the paper cites for the governing equations
and constants is Lane, E. J., 2026: *FrostByte: Mathematical and physical
foundations of the product suite*, Volume I,
<https://doi.org/10.5281/zenodo.22844942>.
