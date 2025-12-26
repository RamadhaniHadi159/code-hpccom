import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Point
import os

# ==================================================
# PATH
# ==================================================
data_path = "/home/hpccom/prosiding/data_done_log.txt"
shp_path = "gadm41_IDN_2.shp"
output_dir = "output_jatim_grid0.25"
os.makedirs(output_dir, exist_ok=True)

# ==================================================
# SHAPEFILE JAWA TIMUR
# ==================================================
gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
jatim = gdf[gdf["NAME_1"] == "Jawa Timur"]

# ==================================================
# DATA CO
# ==================================================
df = pd.read_csv(data_path, delim_whitespace=True)
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df["year"] = df["date"].dt.year

def tentukan_musim(bulan):
    return "hujan" if bulan in [11,12,1,2,3] else "kemarau"

df["musim"] = df["date"].dt.month.apply(tentukan_musim)

# ==================================================
# LOOP
# ==================================================
for (tahun, musim), grup in df.groupby(["year", "musim"]):

    rata = grup.groupby(["lat", "lon"])["log_kadar_co"].mean().reset_index()
    if rata.empty:
        continue

    # ==============================================
    # GRID 0.25°
    # ==============================================
    lon_min, lat_min, lon_max, lat_max = jatim.total_bounds

    lon_grid = np.arange(lon_min, lon_max + 0.25, 0.25)
    lat_grid = np.arange(lat_min, lat_max + 0.25, 0.25)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    zi = griddata(
        (rata["lon"], rata["lat"]),
        rata["log_kadar_co"],
        (lon_mesh, lat_mesh),
        method="linear"
    )

    # ==============================================
    # GRID → POINT
    # ==============================================
    grid_df = pd.DataFrame({
        "lon": lon_mesh.flatten(),
        "lat": lat_mesh.flatten(),
        "co": zi.flatten()
    }).dropna()

    grid_gdf = gpd.GeoDataFrame(
        grid_df,
        geometry=gpd.points_from_xy(grid_df.lon, grid_df.lat),
        crs="EPSG:4326"
    )

    # ==============================================
    # SPATIAL JOIN (INTERSECTS → PENTING)
    # ==============================================
    join = gpd.sjoin(grid_gdf, jatim, how="left", predicate="intersects")

    kab_mean = (
        join.groupby("NAME_2")["co"]
        .mean()
        .reset_index()
    )

    # ==============================================
    # GABUNG KE SEMUA KABUPATEN
    # ==============================================
    jatim_plot = jatim.merge(kab_mean, on="NAME_2", how="left")

    # ==============================================
    # PLOT (SEMUA KABUPATEN TAMPIL)
    # ==============================================
    fig, ax = plt.subplots(figsize=(8,10))

    jatim_plot.plot(
        column="co",
        cmap="plasma",
        linewidth=0.6,
        edgecolor="black",
        legend=True,
        missing_kwds={
            "color": "lightgrey",
            "label": "Tidak ada data"
        },
        ax=ax
    )

    ax.set_title(f"Peta Log(Kadar CO)\nJawa Timur – {musim.capitalize()} {tahun}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/peta_co_jatim_{tahun}_{musim}.png",
        dpi=300
    )
    plt.close()

print("✅ SEMUA KABUPATEN JAWA TIMUR TAMPIL (GRID 0.25°)")
