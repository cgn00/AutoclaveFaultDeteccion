from numpy import NaN
import pandas as pd

path = "D:\\CGN\projects\\AutoclaveFailDeteccion\\data\\Datos\\executions.csv"

data = pd.read_csv(path, sep=",")

rows = len(data.axes[0])

text = "NULL"

for i in range(1, rows, 1):
    if data.loc[i,"StartOpId"] == text:
        data.loc[i,"StartOpId"] = "NaN"
    elif data.loc[i,"EndOperatorId"] == text:
        data.loc[i,"EndOperatorId"] = "NaN"

data.to_csv("D:\\CGN\projects\\AutoclaveFailDeteccion\\data\\Datos\\executionsNaN.csv")