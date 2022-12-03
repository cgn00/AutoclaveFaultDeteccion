import pandas as pd
import numpy as np
import logging
import os

class ExecutionsAnalyzer:
    
    """This class is to explore the executions, agrupate all the executions of a given phase...
    """
    
    def __init__(self, path):
        self._base_directory = path 
        self._duration_directory = '\\Durations\\'
        self._variables_directory = "\\Variables\\"
        self._data_directory = "\\Datas\\"
        self._samples_directory = "\\Samples For Executions Cluster\\"
        self._phases_directory = "\\Datos\\phases_to_analysis.csv"
        self._executions_directory = "Datos\\clean_executions.csv"
        self._executions_to_filter = "Datos\\executions.csv"
        self._criterion = "both"
        self._sequence_directory = ""
        self._phases_by_sequence_directory = ""
        self._base_directory = ""
        
            
    def LoadBaseDirectory(self, path):
        self._base_directory = path
    
    def RemoveIncorrectTime(self):
        """
        This function remove executions with incorrect Start or End Time.
        The Start Time can't happen after the End Time
        and the execution (N-1) can't start before the execution (N),
        or The execution (N-1) can't end after the execution (N)
        """
        
        path_to_read = os.path.join(self._base_directory, self._executions_to_filter)
        path_to_save = os.path.join(self._base_directory, self._executions_directory)
        
        if(os.path.exists(path_to_save)):
            logging.info("Executions with incorrect Times were already removed, there isn't nothing to remove ")
            return
        
        executions = pd.read_csv(path_to_read, sep=',')
        
        correct_executions = executions.apply(lambda x: x if x['StartTime'] > x['EndTime'] else np.NaN, axis=1)
        correct_executions.drop
        print()

