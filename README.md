# FT1 — Infrared Anomaly Detection in AllWISE Survey
### By Arslann | Physics Student, NEDUET Karachi

---

## 🔭 Summary

An automated machine learning pipeline to identify unusual infrared sources in NASA's AllWISE catalog — objects that may be previously uncatalogued protostars, brown dwarfs, or AGN.

**Key Finding:** Identified candidate protostar `J180922.16-042347.2` in the Serpens Molecular Cloud with extreme infrared excess (W2-W3=6.1) not present in the official Spitzer YSO catalog of 2,966 known Serpens objects (Dunham et al. 2015).

---

## 📊 Results

| Finding | Value |
|---|---|
| Total sources surveyed | 60,000 |
| Sky directions covered | 6 (RA: 30°, 90°, 150°, 210°, 270°, 330°) |
| Unusual infrared sources found | 11,409 |
| Tier 1 candidates (W2-W3 > 4.0) | ~2,877 |
| Top candidate | J180922.16-042347.2 |
| Top candidate W2-W3 color | 6.105 |
| SIMBAD match | None |
| Gaia detection | None |
| Spitzer Serpens catalog match | None |

---

## 🌟 Top Candidate

**AllWISE ID:** J180922.16-042347.2  
**RA:** 272.342355°  
**Dec:** -4.396445°  
**W1-W2:** 1.314  
**W2-W3:** 6.105  
**Region:** Serpens Molecular Cloud (436 light years)  

### Evidence this is a genuine uncatalogued protostar:
- ✅ Extreme infrared excess W2-W3=6.1 (normal stars ~ 0)
- ✅ Completely invisible in optical light (DSS2/Aladin)
- ✅ Undetected by Gaia (too dust-obscured)
- ✅ No SIMBAD classification exists
- ✅ Not in Spitzer Serpens YSO catalog (Dunham+ 2015, 2966 objects)
- ✅ Spitzer MIPS observed the region — object was genuinely overlooked
- ✅ Brightens progressively from W1 → W2 → W3 → W4 (classic YSO signature)

**Most likely classification:** Class 0/I protostar — a star in the earliest stage of formation, still embedded in its birth cloud.

---

## 🛠️ Pipeline

```
AllWISE Catalog (VizieR)
        ↓
Infrared Color Selection (W1-W2, W2-W3, W3-W4 cuts)
        ↓
Isolation Forest Anomaly Ranking
        ↓
Random Forest ML Classification
        ↓
SIMBAD Cross-match
        ↓
Gaia Cross-match
        ↓
Spitzer Catalog Cross-match
        ↓
WISE Image Verification
        ↓
Candidate J180922.16-042347.2 identified
```

---

## 📁 Files

| File | Description |
|---|---|
| `discover_hidden_objects.py` | Main NEOWISE discovery pipeline |
| `allwise_spread.csv` | 60,000 AllWISE sources across 6 sky directions |
| `hidden_objects.csv` | 46,587 color-selected unusual sources |
| `tier1_spread.csv` | Top tier unusual candidates |
| `anomaly_ranked.csv` | Top 20 most anomalous by Isolation Forest |
| `ml_classified.csv` | Random Forest predictions |
| `gaia_matches.csv` | Gaia cross-match results |
| `simbad_checked.csv` | SIMBAD cross-match results |
| `spread_plot.png` | Color-color diagram of all candidates |
| `good_lc.png` | NEOWISE light curves |
| `candidate_images/` | 2MASS infrared images of top candidates |
| `classifier_model.pt` | Trained PyTorch classifier |

---


## 🛠️ How to Run

```bash
git clone https://github.com/Azfar1236/FT1.git
cd FT1
pip install astroquery astropy matplotlib pandas torch scikit-learn PyWavelets
python3 discover_hidden_objects.py --ra-min 0 --ra-max 360 --dec-min -60 --dec-max 60 --region-name real_sky --model classifier_model.pt
```

---

## 📚 References

- Wright et al. 2010 — WISE All-Sky Survey
- Dunham et al. 2015 — Spitzer YSO catalog of Serpens (J/ApJS/220/11)
- Cutri et al. 2013 — AllWISE Data Release

---

## 📬 Contact

**SYEDA ZOHA ALI** | Physics Student | NEDUET Karachi, Pakistan  
GitHub: github.com/Azfar1236/FT1
