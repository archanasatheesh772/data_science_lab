import pandas as pd
data = {"Name":["paul","Raj","Reena","Anu","Amal"],
      "Age":[75,34,25,16,15],
      "city":["kottayam","alappuzha","erk","idukki","kozhikode"]}
df=pd.DataFrame(data)
df.to_csv("student.csv",index=False)
print("csv file created")