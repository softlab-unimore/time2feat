import pandas as pd
from glob import glob


results = []
for f in glob("./output/**/*.csv", recursive=True):
    _, _, seed, ds = f.split("/")
    ds, prop = ds.rsplit(".", 1)[0].split("_")

    results.append(pd.melt(pd.read_csv(f), id_vars=["Unnamed: 0"], var_name="measure"))

    results[-1]["dataset"] = ds
    results[-1]["seed"] = seed
    results[-1]["proportion"] = prop

results = pd.concat(results)