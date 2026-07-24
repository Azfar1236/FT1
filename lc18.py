import warnings,pandas as pd,numpy as np,matplotlib
warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
from astropy import units as u
Irsa.TIMEOUT=120
df=pd.read_csv("good_candidates.csv",header=None)
df[2]=pd.to_numeric(df[2],errors="coerce")
df[3]=pd.to_numeric(df[3],errors="coerce")
cands=[(float(r[2]),float(r[3]),str(r[5])) for r in df.values]
n=len(cands)
print(str(n)+" candidates found")
fig,axes=plt.subplots(5,4,figsize=(16,20),facecolor="k")
axes=axes.flatten()
for i,c in enumerate(cands):
    ax=axes[i]
    ax.set_facecolor("#161b22")
    try:
        t=Irsa.query_region(SkyCoord(ra=c[0]*u.deg,dec=c[1]*u.deg),catalog="neowiser_p1bs_psd",spatial="Cone",radius=5*u.arcsec)
        d=t.to_pandas()
        yr=2000+(d["mjd"].values-51544.5)/365.25
        g=d["w1mpro"].notna()
        ax.scatter(yr[g],d["w1mpro"].values[g],c="cyan",s=6,alpha=0.6)
        ax.invert_yaxis()
        print("Done "+str(i+1)+"/"+str(n))
    except Exception as e:
        ax.text(0.5,0.5,"failed",color="red",ha="center",va="center",transform=ax.transAxes)
        print("Failed "+str(i+1))
    ax.set_title("RA="+str(round(c[0],2))+" "+c[2],color="w",fontsize=6)
    ax.tick_params(colors="grey",labelsize=6)
for j in range(n,20):
    axes[j].set_visible(False)
plt.tight_layout()
plt.savefig("/workspace/FT1/good_lc.png",dpi=120,bbox_inches="tight",facecolor="k")
print("Saved good_lc.png")
