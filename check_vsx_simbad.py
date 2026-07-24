#!/usr/bin/env python3
"""
check_vsx_simbad.py

Cross-checks candidates from discoveries_map_top_candidates.csv (or any CSV
with ra_centroid/dec_centroid columns) against two catalogs your pipeline
hasn't checked yet:

  - VSX  (AAVSO's International Variable Star Index - the actual catalog of
          known variable stars, via VizieR table B/vsx/vsx)
  - SIMBAD (general astronomical object database)

A candidate is only a genuinely interesting "still unlisted" lead if it's
absent from AllWISE (already checked by your pipeline) AND VSX AND SIMBAD.

Usage
-----
    pip install astroquery pandas --break-system-packages
    python3 check_vsx_simbad.py discoveries_map_top_candidates.csv
"""

import sys
import time
import pandas as pd
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

MATCH_RADIUS_ARCSEC = 5.0


def check_simbad(ra, dec):
    try:
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        result = Simbad.query_region(coord, radius=MATCH_RADIUS_ARCSEC * u.arcsec)
        return result is not None and len(result) > 0
    except Exception as e:
        return None  # inconclusive - network hiccup, don't count as "known"


def check_vsx(vizier, ra, dec):
    try:
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        result = vizier.query_region(coord, radius=MATCH_RADIUS_ARCSEC * u.arcsec,
                                      catalog="B/vsx/vsx")
        return result is not None and len(result) > 0
    except Exception as e:
        return None


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "discoveries_map_top_candidates.csv"
    out_path = csv_path.replace(".csv", "_vsx_simbad_checked.csv")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} candidates from {csv_path}")

    vizier = Vizier(columns=["*"])

    in_vsx, in_simbad = [], []
    for i, row in df.iterrows():
        ra, dec = row["ra_centroid"], row["dec_centroid"]

        vsx_hit = check_vsx(vizier, ra, dec)
        simbad_hit = check_simbad(ra, dec)

        in_vsx.append(vsx_hit)
        in_simbad.append(simbad_hit)

        status = []
        status.append("VSX" if vsx_hit else ("VSX?" if vsx_hit is None else "-"))
        status.append("SIMBAD" if simbad_hit else ("SIMBAD?" if simbad_hit is None else "-"))
        print(f"[{i+1}/{len(df)}] RA={ra:.5f} Dec={dec:.5f}  {' '.join(status)}")

        time.sleep(0.5)  # be polite to the catalog services

    df["in_vsx"] = in_vsx
    df["in_simbad"] = in_simbad
    df["still_unlisted"] = [
        (v is False) and (s is False) for v, s in zip(in_vsx, in_simbad)
    ]

    df.to_csv(out_path, index=False)

    n_unlisted = df["still_unlisted"].sum()
    print(f"\nSaved: {out_path}")
    print(f"{n_unlisted} of {len(df)} candidates absent from AllWISE, VSX, "
          f"AND SIMBAD - these are your strongest leads.")
    if n_unlisted > 0:
        print("\nTop unlisted candidates:")
        print(df[df["still_unlisted"]][["ra_centroid", "dec_centroid",
                                         "predicted_class"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
