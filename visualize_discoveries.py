#!/usr/bin/env python3
"""
visualize_discoveries.py

Reads discoveries_allsky.csv (produced by discover_hidden_objects.py) and
makes a sky-map image of every candidate, plus a short text summary. Run
this ON THE POD, next to the CSV file - it has no network dependency,
just matplotlib/pandas reading a local file.

Usage
-----
    pip install matplotlib pandas --break-system-packages
    python3 visualize_discoveries.py discoveries_allsky.csv
"""

import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display on a headless pod
import matplotlib.pyplot as plt


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "discoveries_allsky.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "discoveries_map.png"

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} candidates from {csv_path}")
    print(df["predicted_class"].value_counts())
    print(df["catalog_status"].value_counts())

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                              gridspec_kw={"width_ratios": [2, 1]})

    # ---- Left: sky map, RA/Dec, colored by class, "new" ones highlighted ----
    ax = axes[0]
    known = df[df["catalog_status"] == "known"]
    inconclusive = df[df["catalog_status"] == "inconclusive"]
    new = df[df["catalog_status"] == "new"]

    ax.scatter(known["ra_centroid"], known["dec_centroid"],
               s=3, c="#7a7a7a", alpha=0.3, label=f"Known (AllWISE) [{len(known)}]")
    if len(inconclusive):
        ax.scatter(inconclusive["ra_centroid"], inconclusive["dec_centroid"],
                   s=3, c="#e8b84b", alpha=0.4,
                   label=f"Inconclusive check [{len(inconclusive)}]")
    ax.scatter(new["ra_centroid"], new["dec_centroid"],
               s=14, c="#e8433a", edgecolors="black", linewidths=0.3,
               label=f"NOT in AllWISE (candidates) [{len(new)}]", zorder=5)

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"NEOWISE variable candidates - {len(df)} total")
    ax.legend(loc="upper right", fontsize=8)
    ax.invert_xaxis()  # conventional sky-map orientation

    # ---- Right: breakdown bar chart of predicted_class x catalog_status ----
    ax2 = axes[1]
    pivot = df.groupby(["predicted_class", "catalog_status"]).size().unstack(fill_value=0)
    pivot.plot(kind="bar", stacked=True, ax=ax2,
               color={"known": "#7a7a7a", "inconclusive": "#e8b84b", "new": "#e8433a"})
    ax2.set_title("Class breakdown")
    ax2.set_xlabel("")
    ax2.set_ylabel("count")
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")

    # ---- Also dump just the "new" candidates, sorted by confidence, so you
    # can eyeball the strongest leads first ----
    if len(new):
        prob_col = new.apply(
            lambda r: r[f"prob_{r['predicted_class']}"], axis=1)
        new = new.assign(confidence=prob_col).sort_values(
            "confidence", ascending=False)
        top_path = out_path.replace(".png", "_top_candidates.csv")
        new.head(50).to_csv(top_path, index=False)
        print(f"Saved top 50 highest-confidence new candidates: {top_path}")


if __name__ == "__main__":
    main()
