#!/usr/bin/env python3
"""
plot_light_curves.py

Re-queries NEOWISE for the EXACT coordinates of your top candidates (small
cone search, not a whole tile) and plots brightness (w1mpro) vs time (mjd)
for each one. This is the single most useful sanity check available from
a terminal: a real periodic/transient variable shows a believable pattern
in this plot; noise or an artifact usually doesn't.

This does NOT "confirm" a discovery (that needs independent follow-up
observation - see chat) but it's the strongest evidence you can gather
without a telescope.

Usage
-----
    pip install requests pandas matplotlib --break-system-packages
    python3 plot_light_curves.py discoveries_map_top_candidates_vsx_simbad_checked.csv
"""

import sys
import io
import csv as csv_module
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAP_SYNC_URL = "https://irsa.ipac.caltech.edu/TAP/sync"
NEOWISE_TABLE = "neowiser_p1bs_psd"
CONE_RADIUS_ARCSEC = 3.0


def fetch_light_curve(session, ra, dec):
    radius_deg = CONE_RADIUS_ARCSEC / 3600.0
    adql = f"""
        SELECT mjd, w1mpro, w2mpro
        FROM {NEOWISE_TABLE}
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
        ) = 1
          AND qual_frame > 0
          AND cc_flags = '0000'
        ORDER BY mjd ASC
    """.strip()
    resp = session.get(TAP_SYNC_URL, params={"QUERY": adql, "FORMAT": "csv"},
                        timeout=60)
    resp.raise_for_status()
    rows = list(csv_module.DictReader(io.StringIO(resp.text)))
    mjd = [float(r["mjd"]) for r in rows if r["mjd"] and r["w1mpro"]]
    mag = [float(r["w1mpro"]) for r in rows if r["mjd"] and r["w1mpro"]]
    return mjd, mag


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "discoveries_map_top_candidates.csv"
    n_plot = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    df = pd.read_csv(csv_path)
    if "still_unlisted" in df.columns:
        df = df[df["still_unlisted"]]
    df = df.head(n_plot).reset_index(drop=True)
    print(f"Plotting light curves for {len(df)} candidates...")

    session = requests.Session()
    ncols = 3
    nrows = (len(df) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                              squeeze=False)

    for i, row in df.iterrows():
        ax = axes[i // ncols][i % ncols]
        ra, dec = row["ra_centroid"], row["dec_centroid"]
        try:
            mjd, mag = fetch_light_curve(session, ra, dec)
        except Exception as e:
            ax.set_title(f"RA={ra:.3f} Dec={dec:.3f}\nquery failed: {e}",
                         fontsize=8, color="red")
            continue

        print(f"  [{i+1}/{len(df)}] RA={ra:.4f} Dec={dec:.4f}: "
              f"{len(mjd)} points")

        if not mjd:
            ax.set_title(f"RA={ra:.3f} Dec={dec:.3f}\nno points returned",
                         fontsize=8, color="orange")
            continue

        ax.scatter(mjd, mag, s=10, alpha=0.7)
        ax.invert_yaxis()  # brighter = lower magnitude, conventional orientation
        ax.set_title(f"RA={ra:.3f} Dec={dec:.3f}\n"
                     f"{row.get('predicted_class', '?')}, {len(mjd)} pts",
                     fontsize=8)
        ax.set_xlabel("MJD", fontsize=7)
        ax.set_ylabel("W1 mag", fontsize=7)
        ax.tick_params(labelsize=6)

    # hide unused subplot slots
    for j in range(len(df), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    plt.tight_layout()
    out_path = "light_curves.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    print("\nWhat to look for: a real periodic variable shows a repeating "
          "up-down pattern; a real transient shows one clear dip/spike "
          "then flat; noise looks scattered with no pattern; a single "
          "isolated point far from the rest across a short time window "
          "often means an asteroid, not a persistent star.")


if __name__ == "__main__":
    main()
