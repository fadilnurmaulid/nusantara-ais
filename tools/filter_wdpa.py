from pathlib import Path
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "raw"
    / "WDPA"
    / "WDPA_Jul2026_Public_shp"
    / "WDPA_Jul2026_Public_shp_0"
    / "WDPA_Jul2026_Public_shp-polygons.shp"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "wdpa_indonesia.gpkg"
)

print("Loading WDPA...")

wdpa = gpd.read_file(INPUT)

print("Filtering Indonesia...")

wdpa = wdpa[
    (wdpa["ISO3"] == "IDN") |
    (wdpa["PRNT_ISO3"] == "IDN")
].copy()

wdpa = wdpa.to_crs(4326)

wdpa.to_file(
    OUTPUT,
    driver="GPKG"
)

print("\nDone")
print("Polygon :", len(wdpa))
print("Saved :", OUTPUT)