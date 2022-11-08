from numpy import NaN
import pandas as pd

path = "D:\\CGN\projects\\AutoclaveFailDeteccion\\data\\Datos\\executions.csv"

data = pd.read_csv(path, sep=",")
data = data.fillna(0)
data = data.astype({'StartOpId':'int32', 'EndOperatorId':'int32'})

data.to_csv("D:\\CGN\projects\\AutoclaveFailDeteccion\\data\\Datos\\executions_new.csv", index=False)