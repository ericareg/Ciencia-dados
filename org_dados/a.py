import pandas as pd

df = pd.read_csv("proc_data/saida_merged_2016.csv")
a = df['municipio_nome'].value_counts()

print(a)