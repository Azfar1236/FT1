#!/usr/bin/env python3
"""
discover_hidden_objects.py

End-to-end pipeline to discover previously-uncataloged variable sources in
NEOWISE data, following the same overall approach as Paz (2024, VARnet)
and Paz et al. (2026, VarWISE): pull raw catalog rows, spatially cluster
them with no prior source list, classify each discovered cluster's light
curve, and cross-check against known catalogs to see which candidates are
genuinely new. Nothing in this script ever requests an image, cutout, or
FITS file - every network call returns plain numeric/text catalog rows.

READ THIS BEFORE TREATING THIS AS "IDENTICAL" TO PAZ'S PIPELINE
------------------------------------------------------------------
No code release exists for either of his papers (checked directly - no
GitHub repo under his name, VARnet, or VarWISE). Two pieces are therefore
substitutes, not reproductions, clearly marked where they occur:

  1. Classifier internals (train_classifier.py): uses a non-uniform DFT
     in place of his undisclosed "Finite-Embedding Fourier Transform,"
     and a standard Daubechies wavelet transform. Same problem, same
     general architecture shape (wavelet + Fourier feature + CNN), not
     his exact formulas/hyperparameters.
  2. DBSCAN parameters (eps, minPts below): his papers don't publish the
     exact values used; the ones here are reasonable defaults for WISE's
     ~6" beam and typical apparition density, not confirmed to match his.

What IS confirmed and matched exactly: the basic apparition fields
(ra, dec, mjd, w1mpro, w2mpro), the use of DBSCAN for source-agnostic
discovery (Paz et al. 2026 report 456,124,525 clusters from ~98.5 billion
of the ~188.9 billion total rows), and the 3-class taxonomy his production
pipeline actually uses (Null / Transient / Continuous Variable).

Pipeline stages
---------------
  1. Query raw apparitions in a sky tile via IRSA TAP async (catalog rows
     only).
  2. DBSCAN-cluster the (ra, dec) point-cloud - no prior source list, so
     clusters can correspond to anything, cataloged or not.
  3. Build a light curve (mjd, w1mpro) per cluster and classify it with
     the trained model from train_classifier.py.
  4. Drop "null" (non-variable) classifications - matches Paz et al.
     (2026)'s pipeline, which discards nulls before further analysis.
  5. For remaining candidates, cross-match against the AllWISE source
     catalog (a small cone-search query, still just numbers) to flag
     whether each is already cataloged or genuinely new - this is the
     step that actually answers "is this hidden/undiscovered."

Usage
-----
    pip install requests numpy scikit-learn torch pywavelets --break-system-packages

    # First train the classifier (see train_classifier.py), then:
    python3 discover_hidden_objects.py \\
        --ra-min 180 --ra-max 190 --dec-min 8 --dec-max 16 \\
        --region-name virgo_core --model classifier_model.pt

Resuming
--------
Checkpoints per Dec band, same pattern as the other scripts - re-run the
same command to resume after a dropped connection.
"""

import argparse
import csv
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import requests
import torch
from sklearn.cluster import DBSCAN

from train_classifier import (VARnetStyleCNN, CLASSES,
                               resample_to_grid, ndft_magnitude,
                               wavelet_features, GRID_LEN)


def generate_tiles(tile_size_deg: float):
    """Tiles the ENTIRE sky into RA x Dec patches: fixed-height Dec bands,
    with RA width scaled by 1/cos(dec) so polar tiles aren't absurdly thin
    slivers compared to equatorial ones. This is what makes --all-sky
    actually cover the whole sky instead of one RA/Dec box."""
    dec = -90.0
    while dec < 90.0:
        dec_hi = min(dec + tile_size_deg, 90.0)
        dec_mid = (dec + dec_hi) / 2.0
        cos_factor = max(math.cos(math.radians(dec_mid)), 0.05)
        ra_width = min(tile_size_deg / cos_factor, 360.0)
        n_ra_tiles = max(int(math.ceil(360.0 / ra_width)), 1)
        actual_ra_width = 360.0 / n_ra_tiles

        ra = 0.0
        for _ in range(n_ra_tiles):
            ra_hi = ra + actual_ra_width
            yield (ra, ra_hi, dec, dec_hi)
            ra = ra_hi
        dec = dec_hi

TAP_SYNC_URL = "https://irsa.ipac.caltech.edu/TAP/sync"
TAP_ASYNC_URL = "https://irsa.ipac.caltech.edu/TAP/async"
NEOWISE_TABLE = "neowiser_p1bs_psd"
ALLWISE_TABLE = "allwise_p3as_psd"   # for known-object cross-match

MAX_RETRIES = 5
BACKOFF_BASE = 2.0
REQUEST_TIMEOUT = 60
POLL_INTERVAL = 10
JOB_MAX_WAIT = 3600
MAXREC = 3_000_000
PAGE_DEC_HEIGHT = 0.5

DBSCAN_EPS_DEG = 3.0 / 3600.0   # ~3 arcsec - not confirmed to match Paz's value
DBSCAN_MIN_SAMPLES = 8          # not confirmed to match Paz's value

KNOWN_OBJECT_MATCH_RADIUS_ARCSEC = 5.0
MIN_APPARITIONS_FOR_CLASSIFICATION = 8


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"done_pages": {}}


def save_checkpoint(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def build_neowise_adql(ra_min, ra_max, dec_min, dec_max) -> str:
    return f"""
        SELECT ra, dec, mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro,
               cc_flags, qual_frame
        FROM {NEOWISE_TABLE}
        WHERE ra BETWEEN {ra_min} AND {ra_max}
          AND dec BETWEEN {dec_min} AND {dec_max}
          AND qual_frame > 0
          AND cc_flags = '0000'
    """.strip()


def submit_async_job(session: requests.Session, adql: str) -> str:
    resp = session.post(TAP_ASYNC_URL, data={
        "REQUEST": "doQuery", "LANG": "ADQL", "QUERY": adql,
        "FORMAT": "csv", "MAXREC": MAXREC, "PHASE": "RUN",
    }, timeout=REQUEST_TIMEOUT, allow_redirects=False)
    resp.raise_for_status()
    job_url = resp.headers.get("Location")
    if not job_url and resp.status_code in (301, 302, 303) and resp.next:
        job_url = resp.next.url
    if not job_url:
        raise RuntimeError(f"No job URL returned (status {resp.status_code})")
    return job_url


def poll_job(session: requests.Session, job_url: str, tile_name: str) -> str:
    waited = 0
    while waited < JOB_MAX_WAIT:
        resp = session.get(f"{job_url}/phase", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        phase = resp.text.strip()
        if phase in ("COMPLETED", "ERROR", "ABORTED"):
            return phase
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
    raise TimeoutError(f"{tile_name}: job did not complete within {JOB_MAX_WAIT}s")


def fetch_job_results(session: requests.Session, job_url: str) -> list:
    resp = session.get(f"{job_url}/results/result", timeout=REQUEST_TIMEOUT,
                        stream=True)
    resp.raise_for_status()
    # Explicitly decode each chunk as UTF-8 bytes -> str. requests'
    # decode_unicode=True can inconsistently yield raw bytes instead of
    # str depending on the response's declared encoding, which then fails
    # when written into a text buffer ("string argument expected, got
    # bytes"). Decoding manually here avoids that failure mode entirely.
    text_buf = io.StringIO()
    for chunk in resp.iter_content(chunk_size=64 * 1024):
        if chunk:
            text_buf.write(chunk.decode("utf-8", errors="replace"))
    text_buf.seek(0)
    return list(csv.DictReader(text_buf))


def query_with_retry(session, submit_fn, tile_name):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return submit_fn()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.HTTPError,
                RuntimeError, TimeoutError) as e:
            last_exc = e
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            log(f"  {tile_name}: attempt {attempt}/{MAX_RETRIES} failed "
                f"({type(e).__name__}: {e}) - retrying in {wait:.0f}s")
            time.sleep(wait)
    raise last_exc


def query_tile_async(session, ra_min, ra_max, dec_min, dec_max, tile_name):
    adql = build_neowise_adql(ra_min, ra_max, dec_min, dec_max)

    def do_query():
        job_url = submit_async_job(session, adql)
        phase = poll_job(session, job_url, tile_name)
        if phase != "COMPLETED":
            raise RuntimeError(f"job ended in phase {phase}")
        return fetch_job_results(session, job_url)

    return query_with_retry(session, do_query, tile_name)


def is_known_object(session, ra: float, dec: float):
    """Small cone-search against AllWISE (catalog rows only) to check if
    a cluster centroid already has a cataloged counterpart. Returns True/
    False, or None if the check itself failed (inconclusive)."""
    radius_deg = KNOWN_OBJECT_MATCH_RADIUS_ARCSEC / 3600.0
    adql = f"""
        SELECT TOP 1 designation
        FROM {ALLWISE_TABLE}
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
        ) = 1
    """.strip()
    try:
        resp = session.get(TAP_SYNC_URL, params={"QUERY": adql, "FORMAT": "csv"},
                            timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        rows = list(csv.DictReader(io.StringIO(resp.text)))
        return len(rows) > 0
    except Exception as e:
        log(f"    known-object check failed ({e}) - marking as inconclusive")
        return None


def classify_cluster(model, device, mjd: np.ndarray, mag: np.ndarray) -> dict:
    """Builds the same [raw, NDFT, wavelet] feature stack used in
    training, runs it through the classifier, returns class probabilities."""
    grid_mags = resample_to_grid(mjd, mag)
    grid_mags_norm = (grid_mags - grid_mags.mean()) / (grid_mags.std() + 1e-6)

    ndft_mag = ndft_magnitude(mjd, mag)
    ndft_mag_padded = np.interp(np.linspace(0, 1, GRID_LEN),
                                 np.linspace(0, 1, len(ndft_mag)), ndft_mag)

    wave_feats = wavelet_features(grid_mags_norm)

    channels = np.vstack([
        grid_mags_norm[None, :],
        ndft_mag_padded[None, :],
        wave_feats,
    ]).astype(np.float32)

    x = torch.from_numpy(channels).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


def process_tile(rows: list, model, device, session) -> list:
    if not rows:
        return []

    coords = np.array([[float(r["ra"]), float(r["dec"])] for r in rows])
    labels = DBSCAN(eps=DBSCAN_EPS_DEG, min_samples=DBSCAN_MIN_SAMPLES,
                     metric="euclidean").fit_predict(coords)

    results = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # noise, not a real source

        cluster_rows = [r for r, lbl in zip(rows, labels) if lbl == cluster_id]
        if len(cluster_rows) < MIN_APPARITIONS_FOR_CLASSIFICATION:
            continue

        mjds, mags = [], []
        for r in cluster_rows:
            try:
                mjds.append(float(r["mjd"]))
                mags.append(float(r["w1mpro"]))
            except (ValueError, KeyError):
                continue
        if len(mjds) < MIN_APPARITIONS_FOR_CLASSIFICATION:
            continue

        mjds_arr, mags_arr = np.array(mjds), np.array(mags)
        order = np.argsort(mjds_arr)
        mjds_arr, mags_arr = mjds_arr[order], mags_arr[order]

        probs = classify_cluster(model, device, mjds_arr, mags_arr)
        top_class = max(probs, key=probs.get)

        if top_class == "null":
            continue  # discard non-variable, matches Paz et al. 2026

        ras = np.array([float(r["ra"]) for r in cluster_rows])
        decs = np.array([float(r["dec"]) for r in cluster_rows])
        ra_c, dec_c = float(ras.mean()), float(decs.mean())

        known = is_known_object(session, ra_c, dec_c)
        known_status = "known" if known else ("new" if known is False else "inconclusive")

        results.append({
            "ra_centroid": ra_c,
            "dec_centroid": dec_c,
            "n_apparitions": len(cluster_rows),
            "predicted_class": top_class,
            "prob_null": probs["null"],
            "prob_transient": probs["transient"],
            "prob_continuous_variable": probs["continuous_variable"],
            "catalog_status": known_status,
        })

    return results


def build_tile_list(args):
    """Returns a list of (ra_min, ra_max, dec_min, dec_max, tile_name)
    covering either the whole sky (--all-sky) or one custom box paged
    into Dec bands (the original --ra-min/--ra-max/... mode)."""
    if args.all_sky:
        all_tiles = list(generate_tiles(args.tile_size_deg))
        total = len(all_tiles)
        tile_end = args.tile_end if args.tile_end is not None else total
        selected = list(enumerate(all_tiles))[args.tile_start:tile_end]
        log(f"ALL-SKY mode: sky tiled into {total} tiles at "
            f"{args.tile_size_deg} deg. This run covers tiles "
            f"[{args.tile_start}:{tile_end}] ({len(selected)} tiles).")
        return [(ra_min, ra_max, dec_min, dec_max, f"tile{idx}")
                for idx, (ra_min, ra_max, dec_min, dec_max) in selected]
    else:
        ra_min, ra_max = args.ra_min, args.ra_max
        dec_min, dec_max = args.dec_min, args.dec_max
        n_bands = int(np.ceil((dec_max - dec_min) / PAGE_DEC_HEIGHT))
        log(f"CUSTOM REGION mode: RA [{ra_min},{ra_max}] "
            f"Dec [{dec_min},{dec_max}] split into {n_bands} Dec bands. "
            f"NOTE: this covers ONLY this box, not the whole sky - "
            f"use --all-sky for that.")
        tiles = []
        for i in range(n_bands):
            band_dec_min = dec_min + i * PAGE_DEC_HEIGHT
            band_dec_max = min(dec_min + (i + 1) * PAGE_DEC_HEIGHT, dec_max)
            tiles.append((ra_min, ra_max, band_dec_min, band_dec_max,
                          f"{args.region_name}_band{i}"))
        return tiles


def run(tiles, region_name, model_path, checkpoint_path, results_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    checkpoint = torch.load(model_path, map_location=device)
    model = VARnetStyleCNN(n_channels=checkpoint["n_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log(f"Loaded classifier from {model_path}")

    state = load_checkpoint(checkpoint_path)
    session = requests.Session()

    results_file_exists = os.path.exists(results_path)
    results_f = open(results_path, "a", newline="")
    fieldnames = ["region", "page", "ra_centroid", "dec_centroid",
                  "n_apparitions", "predicted_class", "prob_null",
                  "prob_transient", "prob_continuous_variable",
                  "catalog_status"]
    writer = csv.DictWriter(results_f, fieldnames=fieldnames)
    if not results_file_exists:
        writer.writeheader()

    try:
        for i, (ra_min, ra_max, dec_min, dec_max, page_name) in enumerate(tiles):
            if page_name in state["done_pages"]:
                continue

            log(f"[{i+1}/{len(tiles)}] {page_name}: "
                f"RA [{ra_min:.2f},{ra_max:.2f}] "
                f"Dec [{dec_min:.3f},{dec_max:.3f}]")

            try:
                rows = query_tile_async(session, ra_min, ra_max,
                                         dec_min, dec_max, page_name)
            except Exception as e:
                log(f"  {page_name}: giving up after {MAX_RETRIES} retries "
                    f"({e}). Stopping - just re-run to resume.")
                save_checkpoint(checkpoint_path, state)
                results_f.close()
                sys.exit(1)

            log(f"  {page_name}: {len(rows)} apparitions retrieved")
            candidates = process_tile(rows, model, device, session)
            new_count = sum(1 for c in candidates if c["catalog_status"] == "new")
            log(f"  {page_name}: {len(candidates)} variable candidates "
                f"({new_count} not in AllWISE - possible new discoveries)")

            for c in candidates:
                c["region"] = region_name
                c["page"] = page_name
                writer.writerow(c)
            results_f.flush()

            state["done_pages"][page_name] = True
            save_checkpoint(checkpoint_path, state)

    finally:
        results_f.close()

    log("Selected tile range complete.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-sky", action="store_true",
                     help="Cover the ENTIRE sky, tiled automatically. "
                          "Ignores --ra-min/--ra-max/--dec-min/--dec-max.")
    ap.add_argument("--tile-size-deg", type=float, default=2.0,
                     help="(--all-sky only) Dec-band height in degrees; "
                          "RA width auto-scales by 1/cos(dec)")
    ap.add_argument("--tile-start", type=int, default=0,
                     help="(--all-sky only) first tile index - for "
                          "splitting work across parallel RunPod workers")
    ap.add_argument("--tile-end", type=int, default=None,
                     help="(--all-sky only) last tile index (exclusive); "
                          "default = all remaining tiles")
    ap.add_argument("--ra-min", type=float,
                     help="(custom region mode) required unless --all-sky")
    ap.add_argument("--ra-max", type=float)
    ap.add_argument("--dec-min", type=float)
    ap.add_argument("--dec-max", type=float)
    ap.add_argument("--region-name", default="allsky",
                     help="Label used in checkpoint/results filenames")
    ap.add_argument("--model", default="classifier_model.pt",
                     help="Path to the trained model from train_classifier.py")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--results", default=None)
    args = ap.parse_args()

    if not args.all_sky:
        missing = [name for name, val in
                   [("--ra-min", args.ra_min), ("--ra-max", args.ra_max),
                    ("--dec-min", args.dec_min), ("--dec-max", args.dec_max)]
                   if val is None]
        if missing:
            ap.error(f"{', '.join(missing)} required unless --all-sky is set")

    checkpoint_path = args.checkpoint or f"checkpoint_discover_{args.region_name}.json"
    results_path = args.results or f"discoveries_{args.region_name}.csv"

    tiles = build_tile_list(args)
    run(tiles, args.region_name, args.model, checkpoint_path, results_path)


if __name__ == "__main__":
    main()
