import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
from astropy import units as u
import warnings
warnings.filterwarnings("ignore")

Irsa.TIMEOUT = 120

candidates = [
    {"ra": 53.6991, "dec": -88.4291, "label": "Cand-A"},
    {"ra": 12.9768, "dec": -88.6041, "label": "Cand-B"},
]

fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#0d1117")
fig.suptitle("NEOWISE Raw Light Curves", color="white", fontsize=13)

for i, cand in enumerate(candidates):
    coord = SkyCoord(ra=cand["ra"]*u.deg, dec=cand["dec"]*u.deg)
    print(f"Querying {cand['label']}...")
    t = Irsa.query_region(coord, catalog="neowiser_p1bs_psd",
                          spatial="Cone", radius=5*u.arcsec)
    df = t.to_pandas()
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df)}")
    print(df[["mjd","w1mpro","w2mpro",

