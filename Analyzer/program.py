import pandas as pd
import os
import logging
import sys
#from Analyzer import ExecutionsAnalyzer as ExecAnaly
import Analyzer
import json

try:
    path = os.getcwd().removesuffix('Analyzer') + "baseDirectory.txt"

    base_direct = pd.read_csv(path, header=None).iloc[0].values[0] #reading the base deirectory where is allocated the Data

    analyzer = Analyzer.ExecutionsAnalyzer(base_direct)
    
    analyzer.clean_phases_names_mistakes()

    analyzer.remove_incorrect_time() 

    analyzer.split_sequences()
    


except Exception as err :
    print(f"Unexpected {err=}, {type(err)=}")
    raise
