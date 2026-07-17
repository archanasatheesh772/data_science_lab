import pandas as pd

data = {
    "name": ["paul", "raj", None, "reena"],
    "age": [21, 22, None, None],
    "city": ["kottayam", "erm", "alapuzha", None]
}

df = pd.DataFrame(data)

print("--- DataFrame ---")
print(df)
print("\n--- Missing Values (Boolean Mask) ---")
print(df.isnull())
print("\n--- Total Null Values Per Column ---")
print(df.isnull().sum())