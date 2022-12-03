import pandas as pd
import os
import logging
import sys
#from Analyzer import ExecutionsAnalyzer as ExecAnaly
import Analyzer

var = os.getcwd()

base_direct = pd.read_csv("baseDirectory.txt", header=None).iloc[0].values[0]

analyzer = Analyzer.ExecutionsAnalyzer(base_direct)

analyzer.RemoveIncorrectTime()





