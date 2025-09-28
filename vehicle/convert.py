import requests

urls = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaa.dat",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xab.dat",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xac.dat",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xad.dat",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xae.dat",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaf.dat",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xag.dat",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xah.dat",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xai.dat"
]

columns = [
    "COMPACTNESS",
    "CIRCULARITY",
    "DISTANCE_CIRCULARITY",
    "RADIUS_RATIO",
    "PR.AXIS_ASPECT_RATIO",
    "MAX.LENGTH_ASPECT_RATIO",
    "SCATTER_RATIO",
    "ELONGATEDNESS",
    "PR.AXIS_RECTANGULARITY",
    "MAX.LENGTH_RECTANGULARITY",
    "SCALED_VARIANCE_MAJOR",
    "SCALED_VARIANCE_MINOR",
    "SCALED_VARIANCE_MIN_RAD",
    "SCALED_VARIANCE_MAX_RAD",
    "SKEWNESS_ABOUT_MAJOR",
    "SKEWNESS_ABOUT_MINOR",
    "KURTOSIS_ABOUT_MAJOR",
    "KURTOSIS_ABOUT_MINOR",
    "CLASS"
]

output_file = "vehicle_dataset.txt"
rows = []

for url in urls:
    print(f"Descargando {url}...")
    r = requests.get(url)
    r.raise_for_status()
    for line in r.text.splitlines():
        rows.append(line.strip())

with open(output_file, "w", encoding="utf-8") as f:
    f.write(",".join(columns) + "\n")
    for row in rows:
        f.write(",".join(row.split()) + "\n")

print(f"✅ Guardado en {output_file}")
print(f"Total de filas: {len(rows)}")