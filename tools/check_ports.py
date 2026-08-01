from pathlib import Path
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]

ports = gpd.read_file(
    ROOT / "data/raw/Ports/WPI.shp"
)

print(ports[["PORT_NAME", "COUNTRY"]].head(20))

print()

print("Jumlah negara:")
print(ports["COUNTRY"].value_counts().head(30))