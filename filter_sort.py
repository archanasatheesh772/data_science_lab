import pandas as pd
data={"Name":["paul","Raj","Reena","Anu","Amal"],
      "marks":[75,90,85,60,95]}
df=pd.DataFrame(data)
result=df[df["marks"]>80].sort_values(by="marks",ascending=False)
print(result)