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

    analyzer.remove_executions_with_incorrect_time() 

    analyzer.split_sequences()
    
    for i in range(analyzer._sequences_config.__len__()): #iterate over the sequences loadeds from the phase_configuration.json
        
        analyzer._sequence_directory = analyzer._base_directory + analyzer._data_analysis_directory + '\\'+  analyzer._sequences_config[i]._sequence_name + '\\' # this is the sequence's folder
        analyzer._phases_by_sequence_directory = analyzer._sequences_config[i]._sequence_name + '_phases.csv' #this is the csv where are saved the phases of the actual sequence
        seq_conf = analyzer._sequences_config[i]
        
        for phase_conf in seq_conf._phases_conf: #iterate over each one_phase_config of the actual sequence
            
            analyzer.filter_samples_by_phases(phase_conf)
            analyzer.remove_incorrect_time_series(phase_conf, seq_conf._sequence_name)
            analyzer.calculate_dtw_metrics(phase_conf)
            #analyzer.determinate_epsilon(phase_conf, seq_conf._sequence_name)
            analyzer.label_executions_with_DBSCAN(phase_conf, seq_conf._sequence_name)
            analyzer.plot_time_series(phase_conf, seq_conf._sequence_name)
                
    conf = analyzer._sequences_config        
    
            
except Exception as err :
    print(str(err))
    raise
