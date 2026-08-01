from pathlib import Path
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]

ports = gpd.read_file(
    ROOT / "data/raw/Ports/WPI.shp"
)

ports = ports[ports["COUNTRY"] == "ID"].copy()

print("Jumlah port Indonesia :", len(ports))
print()

print(ports[["PORT_NAME", "LATITUDE", "LONGITUDE"]].head(20))

ports.to_file(
    ROOT / "data/processed/ports_indonesia.gpkg",
    driver="GPKG"
)

print()
print("Saved.")