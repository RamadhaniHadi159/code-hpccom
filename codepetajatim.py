import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from scipy.interpolate import griddata
from matplotlib.colors import BoundaryNorm
from shapely.geometry import Point
import os

# ===============================
#  KONFIGURASI JAWA TIMUR
# ===============================
USE_LOG = False  # false = ghi linier, true = log10(GHI)
N_LEVELS = 15    # jumlah level gradasi warna

RES_DATA = 0.25  # resolusi asli ERA5 (derajat)
RES_PLOT = 0.01  # resolusi visualisasi peta (derajat)


# 1. SHAPEFILE JAWA TIMUR

shapefile = "/home/hpccom/cimon/databataswilayah/gadm41_IDN_2.shp"
gdf = gpd.read_file(shapefile).to_crs(epsg=4326)

jatim = gdf[gdf["NAME_1"] == "Jawa Timur"]
wilayah_bg = gdf[gdf["NAME_1"].isin(["Jawa Timur", "Jawa Tengah", "Bali"])]

# Kabupaten Jawa Timur
kabkot_jatim = jatim.copy()

# Buang Wilayah Berawalan Kota
kabkot_jatim = kabkot_jatim[
    ~kabkot_jatim["NAME_2"].str.startswith("Kota ")
]

# Membersihkan Nama Dengan Awalan Kabupaten
kabkot_jatim["LABEL_NAME"] = (
    kabkot_jatim["NAME_2"].str.title()
    .str.replace("Kota", "", regex=False)
    .str.replace("Kabupaten", "", regex=False)
)

kabkot_jatim["label_point"] = kabkot_jatim.geometry.representative_point()

# 2. DATA GHI (SSRD)
df = pd.read_csv(
    "/home/hpccom/cimon/dataset/ghi/datasetghi_harian.txt",
    sep=r"\s+"
)

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# ===============================
# 3. PERIODE 3 BULAN
# ===============================
def periode_3bulan(bulan):
    if bulan in [1, 2, 3]:
        return "Jan-Mar"
    elif bulan in [4, 5, 6]:
        return "Apr-Jun"
    elif bulan in [7, 8, 9]:
        return "Jul-Sep" 
    else:
        return "Okt-Des"

df["periode"] = df["month"].apply(periode_3bulan)

print(df["periode"].value_counts())

# ===============================
# 5. GLOBAL MIN & MAKS DATASET
# ===============================
if USE_LOG:
    ghi_valid = df["ssrd_wm2"][df["ssrd_wm2"] > 0]
    GLOBAL_VMIN = np.log10(ghi_valid.min())
    GLOBAL_VMAKS = np.log10(ghi_valid.max())
    cbar_label = "log10(GHI) [log(Watt/m⁻²)]"
else:
    GLOBAL_VMIN = df["ssrd_wm2"].min()
    GLOBAL_VMAKS = df["ssrd_wm2"].max()
    cbar_label = "GHI (Watt/m⁻²)"
    
print(f"GHI MIN = {GLOBAL_VMIN:.2f}")    
print(f"GHI MAKS = {GLOBAL_VMAKS:.2f}")

# ===============================
# 6. LEVEL WARNA
# ===============================
levels = np.linspace(GLOBAL_VMIN, GLOBAL_VMAKS, N_LEVELS + 1)
norm = BoundaryNorm(levels, ncolors=256)

# ===============================
# 7. OUTPUT
# ===============================
base_output_dir = "/home/hpccom/cimon/cimon_coba_peta"
output_dir = os.path.join(base_output_dir, "coba_peta_jatimlagi3")
os.makedirs(output_dir, exist_ok=True)

scale_tag = "log" if USE_LOG else "linier"

# ===============================
# 8. LOOP TAHUN & PERIODE
# ===============================
for (tahun, periode), grup in df.groupby(["year", "periode"]):
    
    # ===========================
    # RATA-RATA SPASIAL
    # ===========================
    rata = (
        grup
        .groupby(["latitude", "longitude"])["ssrd_wm2"]
        .mean()  
        .reset_index()
    )
    
    if rata.empty:
        continue

    # ===========================
    # SKALA PLOT
    # ===========================
    if USE_LOG:
        rata = rata[rata["ssrd_wm2"] > 0]
        nilai = np.log10(rata["ssrd_wm2"])
    else:
        nilai = rata["ssrd_wm2"]

    # ===========================
    # GRID KHUSUS JAWA TIMUR
    # ===========================
    minx, miny, maxx, maxy = jatim.total_bounds

    xi = np.arange(minx, maxx + RES_PLOT, RES_PLOT)
    yi = np.arange(miny, maxy + RES_PLOT, RES_PLOT)
    xi, yi = np.meshgrid(xi, yi)    
    
    # ===========================
    # INTERPOLASI
    # ===========================
    
    zi = griddata(
        (rata["longitude"], rata["latitude"]),
        nilai,
        (xi, yi),
        method="linear"
    )
    
    # ===============================
    # MAKS HANYA WILAYAH JAWA TIMUR
    # ===============================
    point = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(xi.ravel(), yi.ravel()),
        crs="EPSG:4326"
    )

    mask = point.within(jatim.geometry.union_all()).values
    zi[~mask.reshape(zi.shape)] = np.nan
    
    # ===========================
    # PLOT
    # ===========================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Wilayah Jateng dan Bali 
    wilayah_bg.boundary.plot(
        ax=ax,
        color="white",
        edgecolor="white",
        zorder=1 # urutan layer kecil=paling bawah dan sebaliknya
    )
    
    # Wialyah Batas Pantai 
    coastline = wilayah_bg.geometry.union_all().boundary
    gpd.GeoSeries(coastline, crs="EPSG:4326").plot(
        ax=ax,
        color="black",
        linewidth=0.5,
        zorder=2
    )
    
    # Wilayah Jatim (GHI)
    cs = ax.contourf(
        xi, yi, zi,
        levels=levels,
        cmap="magma",
        norm=norm,
        zorder=3
    )
    
    # Batas Wilayah Jatim Ditebalkan
    jatim.boundary.plot(
        ax=ax,
        color="black",
        linewidth=0.5,
        zorder=4
    )
        
    # Label Kab/Kota
    for _, row in kabkot_jatim.iterrows():
        ax.text(
            row["label_point"].x,
            row["label_point"].y,
            row["LABEL_NAME"],
            fontsize=2.5,
            ha="center",
            va="center",
            color="black",
            path_effects=[
                pe.withStroke(linewidth=1, foreground="white")
            ],
            zorder=5
        )

    # Fokus Peta
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    
    # GRID KOORDINAT
    ax.set_xticks(np.arange(107, 118, 1))
    ax.set_yticks(np.arange(-10, -4, 1))
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.2,
        alpha=0.6
    )

    # COLORBAR INDOKATOR 
    cbar = fig.colorbar(
        cs,
        ax=ax,
        ticks=levels,
        shrink=0.85,
    )
    
    cbar.set_label(cbar_label, fontsize=10)
    cbar.set_ticks(levels)
    cbar.ax.ticklabel_format(style='plain',useOffset=False)
    cbar.ax.tick_params(labelsize=8)
    
# SIMPAN 
    ax.set_title(
        f"Peta GHI Jawa Timur - {periode} {tahun}",
        fontsize=12
    )
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    plt.savefig(
        f"{output_dir}/peta_ghi_jatim_{tahun}_{periode}_{scale_tag}.png",
        dpi=1000,
        bbox_inches="tight"
    )
    plt.close()

print("✅ Peta GHI Jawa Timur berhasil dibuat")
