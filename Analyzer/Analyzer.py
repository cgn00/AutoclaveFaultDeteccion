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
        self._phases_directory = "Datos\\phases_to_analysis.csv"
        self._executions_directory = "Datos\\clean_executions.csv"
        self._executions_to_filter = "Datos\\executions.csv"
        self._data_analysis = 'Data_Analysis\\'
        self._criterion = "both"
        self._sequence_directory = ""
        self._phases_by_sequence_directory = ""
        
        self._sequences_names = list()
        
        #logger config
        self._logger = logging.getLogger('AnalyzerLogger')
        self._logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler('logger.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.info("------------------------------")
            
            
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


    def SplitSequences(self):
        """Split the Executions to the correspondent sequence,
        generate N folders named as the correspondent sequence
        and inside generate a csv that contains the executions of that sequence 
        """
        
        executions_path = os.path.join(self._base_directory, self._executions_to_filter)
        phases_path = os.path.join(self._base_directory, self._phases_directory)
        
        executions = pd.read_csv(executions_path)
        phases = pd.read_csv(phases_path)
        
        self._sequences_names = executions.loc[:, 'SequenceName'].drop_duplicates() #obtain the names of each sequence in the executions
        
        exec_ids_by_sequence = list() #IDs of each execution gruped by sequences
        phases_by_sequence = list() 
        
        #phases_not_contained = phases #initialy all the phases aren't contained, but if the belog to one sequence the are removed
        phases_ids_containeds = list()
        
        for sequence in self._sequences_names: #itetate on each sequence to assing the executions that belong to each one
            temp_ids = executions.apply(lambda row: row['ExecutionId'] if row['SequenceName'] == sequence else np.nan
                                        , axis=1) #iterate over rows to assing the Executions IDs that belong to that sequence
            
            temp_ids.dropna(inplace=True) # delete the NaN and keep the IDs, this IDs will be used to find the Phases of this sequence
            temp_ids = temp_ids.to_numpy()
            
            #phases_ids_containeds.append(temp_ids)
            
            for id in temp_ids:
                 phases_ids_containeds.append(id)
            
            #temp_phases = pd.DataFrame(columns=phases.columns)
            ''' 
            temp_phases = phases.apply(lambda row: row if row['ExecutionId'] in temp_ids else np.nan
                                       , axis=1) #if the ID of the row it's in temp_ids then the phase is added to this sequence
            
            temp_phases.dropna(inplace=True)# delete the NaN and keep the Phases
            '''
            temp_phases = phases[phases['ExecutionId'].isin(temp_ids)]
            
            #phases_by_sequence.append(temp_phases)
            '''
            phases_not_contained = phases_not_contained.apply(
                lambda row :row if not row['ExecutionId'] in temp_ids else np.nan
                , axis=1) #assing NaN to  the Phases that are included in the Sequence
            
            phases_not_contained.to_frame()
            
            phases_not_contained.dropna(inplace=True)
            '''
            
            folder_to_save = self._base_directory + self._data_analysis + sequence  
            if( os.path.exists(folder_to_save) == False):
                 os.makedirs(folder_to_save)
            
            path_to_save = folder_to_save + '\\' + sequence + '_phases.csv'
            temp_phases.to_csv(path_to_save, index=False, header=phases.columns)
            
            #exec_ids_by_sequence.append(temp_ids)
            
        
        #phases_ids_containeds = np.array(phases_ids_containeds).reshape([1,phases_ids_containeds.__len__()])
        index = phases['ExecutionId'].isin(phases_ids_containeds)
        index_negate = index.apply(lambda row: not row)
        phases_not_contained = phases[index_negate]
        
    
        
        
        
