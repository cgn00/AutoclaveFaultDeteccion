import pandas as pd

path = "D:\\CGN\projects\\AutoclaveFailDeteccion\\data\\Datos\\phases.csv"

data = pd.read_csv(path, sep=",")

print(data)

rows = len(data.axes[0])



count = 0

text_List = []

for i in range(1, rows, 1):
    if(text_List.__contains__(data.loc[i, "Text"])== False):
        text_List.append(data.loc[i, "Text"])


#print(text_List)

text = 'Esterilización '

for i in range(1, rows, 1):

     if data.loc[i, "Text"] == text :
         data.loc[i, "Text"] = "Esterilización"
         count += 1

     elif data.loc[i, "Text"] == "Llenado" :
        data.loc[i, "Text"] = "LLenado"

text_List.clear()
for i in range(1, rows, 1):
    if(text_List.__contains__(data.loc[i, "Text"])== False):
        text_List.append(data.loc[i, "Text"])

#print(text_List)

df = pd.DataFrame(data=text_List)
df.to_csv("D:\\CGN\\projects\\AutoclaveFailDeteccion\\data\\Datos\\phases_names.csv")
#data.to_csv("D:\\CGN\\projects\\AutoclaveFailDeteccion\\data\\Datos\\phases_to_analysis.csv", columns=["EntityId", "ExecutionId", "Time", "Text"], index=False)


