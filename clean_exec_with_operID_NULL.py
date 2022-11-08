from numpy import NaN
import pandas as pd

path = "D:\\cujae\\Programación Avanzada\\Curso 2022\\Laboratorios\\Christian Guzman\\Datos\\executions.csv"

data = pd.read_csv(path, sep=",")
data = data.fillna(0)
data = data.astype({'StartOpId':'int32', 'EndOperatorId':'int32'})

data.to_csv("D:\\cujae\\Programación Avanzada\\Curso 2022\\Laboratorios\\Christian Guzman\\Datos\\executionsNaN.csv",
            index=False)
