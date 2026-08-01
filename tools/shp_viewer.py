import geopandas as gpd

gdf = gpd.read_file("WDPA/WDPA_Jul2026_Public_shp/WDPA_Jul2026_Public_shp_0/WDPA_Jul2026_Public_shp-polygons.shp")

print(gdf.head())
print(gdf.columns)