import pandas as pd

# open ../capfusion_3.parquet, subsample 1000 rows, and save to ../capfusion_3_1000.csv

df = pd.read_parquet("../capsfusion_3.parquet")
df = df.sample(n=1000)
df.to_csv("../capsfusion_3_1000.csv")
