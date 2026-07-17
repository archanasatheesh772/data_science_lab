import pandas as pd

data = {
    "name": ["paul", "raj", None, "reena"],
    "age": [21, 22, None, None],
    "city": ["kottayam", "erm", "alapuzha", None]
}

df = pd.DataFrame(data)
print(df)