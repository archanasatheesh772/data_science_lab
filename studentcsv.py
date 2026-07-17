import pandas as pd
df=pd.read_csv("student.csv")
print("first 2 rows")
print(df.head())
print("last 2 rows")
print(df.tail())