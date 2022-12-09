import pandas as pd
import os
import Analyzer
import logging
import sys
import json
import configuration.phase_conf 

try:
    path = os.getcwd().removesuffix('Analyzer') + "baseDirectory.txt"

    base_direct = pd.read_csv(path, header=None).iloc[0].values[0] #reading the base deirectory where is allocated the Data

    analyzer = Analyzer.executions_analyzer(base_direct)
    
    analyzer.clean_phases_names_mistakes()

    analyzer.remove_incorrect_time() 

    analyzer.split_sequences()
    
    for i in range(analyzer._sequences_config.__len__()):
        analyzer._sequence_directory = analyzer._base_directory + analyzer._data_analysis + '\\'+  analyzer._sequences_config[i]._sequence_name + '\\' # this is the sequence's folder
        #analyzer._phases_by_sequence_directory = os.path.join(sequence_conf, "_phases.csv")
        seq_conf = analyzer._sequences_config[i]
        analyzer._sequences_names
        print(analyzer._sequence_directory)
        
        
    conf = analyzer._sequences_config        
    print(analyzer._sequences_config)
        
        
except Exception as err :
    print(f"Unexpected {err=}, {type(err)=}")
    raise
