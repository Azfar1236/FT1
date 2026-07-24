"""
NEOWISE Light Curve Fetcher — Fixed Version
- Bumped IRSA timeout to 120s with 3 retries + backoff
- Per-visit binning: groups individual exposures (~12-16 per visit) 
  into 6-month windows, computes weighted median + scatter per visit
- Plots both raw (grey dots) and binned (coloured) for honest comparison
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy.io.ascii as ascii_astropy
import warnings
warnings.filterwarnings("ignore")

# ── IRSA config ──────────────────────────────────────────────────────────────
Irsa.TIMEOUT = 120          # seconds (was ~30 by default — that's why it timed out)
NEOWISE_TABLE = "neowiser_p1bs_psd"   # NEOWISE-R single-exposure source table
MAX_RETRIES   = 3
RETRY_BACKOFF = [5, 15, 30]   # seconds between retries

# ── Visit binning config ──────────────────────────────────────────────────────
# NEOWISE revisits every ~180 days; each visit is a cluster of exposures over ~1-2 days
# We bin by grouping MJDs within VISIT_GAP_DAYS of each other
VISIT_GAP_DAYS = 15   # any two observations > 15 days apart = different visit

def fetch_neowise_with_retry(ra, dec, radius_arcsec=5.0, verbose=True):
    """
    Query IRSA for NEOWISE-R single-exposure photometry around (ra, dec).
    Returns an astropy Table or None on failure.
    
    Parameters
    ----------
    ra, dec      : float  — decimal degrees
    radius_arcsec: float  — search cone radius (default 5 arcsec)
    """
    coord  = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
    radius = radius_arcsec * u.arcsec

    for attempt in range(MAX_RETRIES):
        try:
            if verbose:
                print(f"  [IRSA] Query attempt {attempt+1}/{MAX_RETRIES} "
                      f"for RA={ra:.4f}, Dec={dec:.4f} ...")
            
            result = Irsa.query_region(
                coordinates=coord,
                catalog=NEOWISE_TABLE,
                spatial="Cone",
                radius=radius,
                selcols="ra,dec,mjd,w1mpro,w1sigmpro,w2mpro,w2sigmpro,"
                        "qual_frame,saa_sep,moon_masked"
            )
            
            if result is None or len(result) == 0:
                if verbose:
                    print(f"  [IRSA] No rows returned for RA={ra:.4f}")
                return None
            
            if verbose:
                print(f"  [IRSA] OK — {len(result)} raw exposures returned")
            return result

        except Exception as e:
            err_str = str(e)
            if "timeout" in err_str.lower() or "timed out" in err_str.lower():
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF[attempt]
                    print(f"  [IRSA] Timeout on attempt {attempt+1}. "
                          f"Waiting {wait}s then retrying...")
                    time.sleep(wait)
                else:
                    print(f"  [IRSA] All {MAX_RETRIES} attempts timed out. "
                          f"Skipping RA={ra:.4f}")
                    return None
            else:
                print(f"  [IRSA] Non-timeout error: {err_str[:120]}")
                return None


def quality_filter(df):
    """
    Keep only good-quality NEOWISE exposures:
      - qual_frame >= 10      (good frame quality flag)
      - saa_sep  >= 0         (not in South Atlantic Anomaly)
      - moon_masked == 0      (not moon-contaminated)
      - finite magnitudes
    """
    mask = (
        (df["qual_frame"] >= 10) &
        (df["saa_sep"]    >= 0)  &
        (df["moon_masked"] == 0) &
        df["w1mpro"].notna()     &
        df["w2mpro"].notna()
    )
    return df[mask].copy()


def assign_visit_numbers(mjd_array):
    """
    Given a sorted array of MJD values, assign a visit number (0, 1, 2, ...)
    by breaking at gaps larger than VISIT_GAP_DAYS.
    
    Example: MJDs [56700, 56701, 56702, 56889, 56890] → visits [0,0,0,1,1]
    """
    mjd_sorted_idx = np.argsort(mjd_array)
    sorted_mjd     = np.array(mjd_array)[mjd_sorted_idx]
    
    visit_nums        = np.zeros(len(sorted_mjd), dtype=int)
    current_visit     = 0
    for i in range(1, len(sorted_mjd)):
        if sorted_mjd[i] - sorted_mjd[i-1] > VISIT_GAP_DAYS:
            current_visit += 1
        visit_nums[i] = current_visit
    
    # Map back to original order
    result = np.empty(len(mjd_array), dtype=int)
    result[mjd_sorted_idx] = visit_nums
    return result


def bin_by_visit(df):
    """
    For each visit, compute:
      - median MJD of all exposures in the visit
      - weighted median W1 and W2 magnitude
      - uncertainty on the bin (median absolute deviation or sigma clipped std)
      - number of exposures in visit
    
    Returns a DataFrame with one row per visit.
    """
    df = df.copy()
    df["visit"] = assign_visit_numbers(df["mjd"].values)
    
    rows = []
    for v, grp in df.groupby("visit"):
        n = len(grp)
        
        # Weighted mean using inverse-variance weights
        for band, sig_col in [("w1mpro","w1sigmpro"), ("w2mpro","w2sigmpro")]:
            sig = grp[sig_col].values
            mag = grp[band].values
            # Guard against zero/nan sigma
            good = np.isfinite(sig) & (sig > 0) & np.isfinite(mag)
            if good.sum() == 0:
                grp.loc[:, f"{band}_binned"]     = np.nan
                grp.loc[:, f"{band}_binned_err"] = np.nan
                continue
            w   = 1.0 / sig[good]**2
            mu  = np.sum(w * mag[good]) / np.sum(w)
            err = 1.0 / np.sqrt(np.sum(w))
            rows.append({
                "visit"    : v,
                "mjd"      : grp["mjd"].median(),
                "year"     : 2000 + (grp["mjd"].median() - 51544.5) / 365.25,
                "w1"       : mu if band == "w1mpro" else None,
                "w1_err"   : err if band == "w1mpro" else None,
                "w2"       : mu if band == "w2mpro" else None,
                "w2_err"   : err if band == "w2mpro" else None,
                "n_exp"    : n,
            })
            break  # will redo both bands properly below
        
        # Redo cleanly for both bands in one row
        row = {"visit": v, "mjd": grp["mjd"].median(),
               "year": 2000 + (grp["mjd"].median() - 51544.5)/365.25,
               "n_exp": n}
        for band, scol in [("w1mpro","w1sigmpro"), ("w2mpro","w2sigmpro")]:
            sig  = grp[scol].values
            mag  = grp[band].values
            good = np.isfinite(sig) & (sig > 0) & np.isfinite(mag)
            if good.sum() > 0:
                w   = 1.0 / sig[good]**2
                mu  = np.sum(w * mag[good]) / np.sum(w)
                err = 1.0 / np.sqrt(np.sum(w))
            else:
                mu, err = np.nan, np.nan
            short = "w1" if "w1" in band else "w2"
            row[short], row[f"{short}_err"] = mu, err
        rows[-1] = row   # replace the incomplete row we pushed above

    return pd.DataFrame(rows)


def plot_lightcurve_panel(candidates, output_path="neowise_lc_fixed.png"):
    """
    For each candidate dict {ra, dec, label, classifier_class},
    fetch NEOWISE data, bin by visit, and plot a two-panel figure
    (W1 top, W2 bottom) with raw grey scatter + coloured visit bins.
    """
    n = len(candidates)
    fig = plt.figure(figsize=(7 * n, 9), facecolor="#0d1117")
    outer = gridspec.GridSpec(1, n, figure=fig, hspace=0.05, wspace=0.35)

    for col_idx, cand in enumerate(candidates):
        ra, dec   = cand["ra"], cand["dec"]
        label     = cand.get("label", f"RA={ra:.3f}")
        cls_class = cand.get("classifier_class", "?")

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col_idx], hspace=0.08
        )
        ax1 = fig.add_subplot(inner[0])
        ax2 = fig.add_subplot(inner[1], sharex=ax1)

        # ── Fetch ──────────────────────────────────────────────────────────
        raw_table = fetch_neowise_with_retry(ra, dec)
        
        if raw_table is None or len(raw_table) == 0:
            for ax, band in [(ax1,"W1"), (ax2,"W2")]:
                ax.set_facecolor("#161b22")
                ax.text(0.5, 0.5, f"IRSA returned no data\n(timeout or empty)",
                        ha="center", va="center", color="#ff6b6b",
                        transform=ax.transAxes, fontsize=10)
                ax.set_ylabel(f"{band} [mag]", color="#8b949e", fontsize=9)
            ax2.set_xlabel("Year", color="#8b949e", fontsize=9)
            ax1.set_title(f"{label}\nclass: {cls_class}", color="#e6edf3",
                          fontsize=10, pad=6)
            continue

        df_raw = raw_table.to_pandas()
        df_qc  = quality_filter(df_raw)
        
        print(f"  {label}: {len(df_raw)} raw → {len(df_qc)} after QC filter")
        
        if len(df_qc) < 3:
            for ax, band in [(ax1,"W1"), (ax2,"W2")]:
                ax.set_facecolor("#161b22")
                ax.text(0.5, 0.5, f"Too few good exposures\n({len(df_qc)} after QC)",
                        ha="center", va="center", color="#ffa657",
                        transform=ax.transAxes, fontsize=10)
            continue

        df_binned = bin_by_visit(df_qc)
        year_raw  = 2000 + (df_qc["mjd"].values - 51544.5) / 365.25

        for ax, band_raw, band_bin, band_err, color, ylabel in [
            (ax1, "w1mpro", "w1", "w1_err", "#58a6ff", "W1 [3.4 µm mag]"),
            (ax2, "w2mpro", "w2", "w2_err", "#f78166", "W2 [4.6 µm mag]"),
        ]:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

            # Raw exposures — grey noise cloud
            ax.scatter(year_raw, df_qc[band_raw].values,
                       c="#484f58", s=6, alpha=0.45, zorder=1,
                       label=f"Single exp (N={len(df_qc)})")

            # Binned visits — coloured with errorbars
            good_bin = df_binned[band_bin].notna()
            if good_bin.sum() > 0:
                ax.errorbar(
                    df_binned.loc[good_bin, "year"],
                    df_binned.loc[good_bin, band_bin],
                    yerr=df_binned.loc[good_bin, band_err],
                    fmt="o", color=color, ecolor=color,
                    elinewidth=1.2, capsize=3, ms=5, zorder=3,
                    label=f"Visit bin (N={good_bin.sum()} visits)"
                )

            ax.invert_yaxis()   # magnitudes: brighter = lower number = up
            ax.set_ylabel(ylabel, color="#8b949e", fontsize=9)
            ax.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d",
                      labelcolor="#8b949e", loc="best")

            # Annotate amplitude of binned curve
            if good_bin.sum() >= 2:
                amp = (df_binned.loc[good_bin, band_bin].max() - 
                       df_binned.loc[good_bin, band_bin].min())
                ax.annotate(f"Δmag={amp:.2f}", xy=(0.97,0.05),
                            xycoords="axes fraction", ha="right",
                            color=color, fontsize=8)

        ax1.set_title(f"{label}\nclass: {cls_class}  |  "
                      f"{len(df_binned)} visits", 
                      color="#e6edf3", fontsize=10, pad=6)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.set_xlabel("Year", color="#8b949e", fontsize=9)

    fig.suptitle("NEOWISE-R Light Curves — Visit-Binned (fixed timeout + QC)",
                 color="#e6edf3", fontsize=13, y=1.01)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print(f"\n✓ Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    # Your two candidates that returned data last time
    # Add more from your DBSCAN anomaly list here
    candidates = [
        {"ra": 53.699,  "dec":  0.0,   "label": "Cand-A (RA=53.699)",
         "classifier_class": "continuous_variable"},
        {"ra": 12.977,  "dec":  0.0,   "label": "Cand-B (RA=12.977)",
         "classifier_class": "continuous_variable"},
        # Add more candidates from your list, e.g.:
        # {"ra": XX.XXX,  "dec": YY.YYY, "label": "Cand-C", "classifier_class": "dip_transient"},
    ]
    
    print("=" * 60)
    print("NEOWISE Light Curve Pipeline — Fixed Version")
    print(f"Timeout: {Irsa.TIMEOUT}s | Retries: {MAX_RETRIES} | "
          f"Visit gap: {VISIT_GAP_DAYS} days")
    print("=" * 60)
    
    plot_lightcurve_panel(candidates, output_path="/home/claude/neowise_lc_fixed.png")
    
    print("\nInterpretation guide:")
    print("  Grey dots  = individual NEOWISE exposures (noisy)")
    print("  Coloured ○ = visit-averaged magnitude (real signal, if any)")
    print("  Δmag < 0.1 → probably noise, not real variability")
    print("  Δmag > 0.3 → worth cross-matching SIMBAD / VSX manually")
