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
        
        #logger config
        self._logger = logging.getLogger('AnalyzerLogger')
        self._logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler('logger.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
            
            
    def LoadBaseDirectory(self, path):
        """Change the base directory where will be loaded and saved the data 

        Args:
            path (str): path is the new base directory
        """
        self._base_directory = path
    
    def RemoveIncorrectTime(self):
        """
        This function remove executions with incorrect Start or End Time.
        The Start Time can't happen after the End Time
        and the execution (N-1) can't start before the execution (N),
        or The execution (N-1) can't end after the execution (N)
        It's the equivalent CleanExecutionsCSV function of the analysis class in C#
        """
        
        path_to_read = os.path.join(self._base_directory, self._executions_to_filter)
        path_to_save = os.path.join(self._base_directory, self._executions_directory)
        
        if(os.path.exists(path_to_save)):
            self._logger.info("Executions with incorrect Times were already removed, there isn't nothing to remove ")
            
            return
        
        executions = pd.read_csv(path_to_read, sep=',')
        
        correct_exc = pd.DataFrame(columns=executions.columns)
        
        if(executions.loc[0, 'StartTime'] < executions.loc[0, 'EndTime']):
            correct_exc.loc[len(correct_exc)] = executions.loc[0, :]
        
        rows_count = executions.shape[0]
        
        for i in range(1, rows_count, 1):
            if(executions.loc[i, 'StartTime'] >= correct_exc.loc[len(correct_exc) - 1, 'EndTime'] and executions.loc[i, 'StartTime'] < executions.loc[i,'EndTime']):
                #the atual execution has a correct start and end times
                correct_exc.loc[len(correct_exc)] = executions.loc[i, :] #add all the columns of the row
        
        correct_exc.to_csv(path_to_save, header=True, index=False)
        
        self._logger.warning(f"Executions.Count = {len(executions)} \nCorrect_executions.count = {len(correct_exc)}")

