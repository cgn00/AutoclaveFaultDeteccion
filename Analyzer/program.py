import pandas as pd
import os
import logging
import sys
#from Analyzer import ExecutionsAnalyzer as ExecAnaly
import Analyzer

try:
    path = os.getcwd().removesuffix('Analyzer') + "baseDirectory.txt"

    base_direct = pd.read_csv(path, header=None).iloc[0].values[0]

    analyzer = Analyzer.ExecutionsAnalyzer(base_direct)

    analyzer.RemoveIncorrectTime() 

    analyzer.SplitSequences()

except Exception as err :
    print(f"Unexpected {err=}, {type(err)=}")
    raise
