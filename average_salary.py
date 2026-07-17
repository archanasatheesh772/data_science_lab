import pandas as pd
data={"dept":["mca","cse","mca","ECE"],
      "empname":["abc","def","ghi","jkl"],
      "salary":[20000,35000,56000,30000]}
df=pd.DataFrame(data)
print(df)
print("Average salary of each department")
avg=df.groupby("dept")["salary"].mean()
print(avg)