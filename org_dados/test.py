import pandas

df = pandas.read_csv("simam_data/dados_2016.csv")

shape = df.shape

shape_total = shape[0] * shape[1]

print(f"{shape_total:,}")