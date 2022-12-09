import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import json
from configuration import phase_conf as conf

class executions_analyzer:
    
    """This class is to explore the executions, agrupate all the executions by a given phase...
    """
    
    def __init__(self, path):
        self._base_directory = path 
        self._duration_directory = '\\Durations\\'
        self._variables_directory = "\\Variables\\"
        self._data_directory = "\\Datas\\"
        self._samples_directory = "\\Samples For Executions Cluster\\"
        self._phases_with_mistakes_in_name = 'Datos\\phases.csv'    # the phases: Esterilizacion an Llenado appear with difirents mistakes in the name
        self._phases_directory = "Datos\\phases_to_analysis.csv"
        self._executions_directory = "Datos\\clean_executions.csv"
        self._executions_to_filter = "Datos\\executions.csv"
        self._data_analysis = 'Data_Analysis'
        self._phase_configuration_directory = 'Analyzer\\configuration\\phase_configuration.json'
        self._criterion = "both"
        self._date_time_format = "%Y-%m-%d %H:%M:%S.%f"
        self._sequence_directory = ''
        self._phases_by_sequence_directory = ''
                
        route = os.path.join(self._base_directory, self._phase_configuration_directory)
        #self._phase_config = json.load(open(route)) #load the Phase Configuration saved as .json
        dic = json.load(open(route))
        self._phases_config = conf.phases_config(dic)
                
        self._sequences_names = list()
        
        #logger config
        self._logger = logging.getLogger('AnalyzerLogger')
        self._logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler('logger.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.info("--------------------New Execution of the code-------------------")
            
            
    def load_base_directory(self, path):
        """Change the base directory where will be loaded and saved the data 

        Args:
            path (str): path is the new base directory
        """
        self._base_directory = path
    
    
    def clean_phases_names_mistakes(self):
        """This function renames the phases:
        'Esterilización ' to 'Esterilización' (whitout space) and
        'LLenado' to 'Llenado'
        """
        path_to_save = os.path.join(self._base_directory, self._phases_directory)
        if(os.path.exists):
            self._logger.info("phases_to_analysis.csv allready exists, nothing to to in clean_phases_names_mistakes()")
            return
        
        path = os.path.join(self._base_directory, self._phases_with_mistakes_in_name)
        
        data = pd.read_csv(path, sep=",")

        rows = len(data.axes[0])

        count = 0

        text_List = []

        for i in range(1, rows, 1):
            if(text_List.__contains__(data.loc[i, "Text"])== False):
                text_List.append(data.loc[i, "Text"])

        text = 'Esterilización '

        for i in range(1, rows, 1):

            if data.loc[i, "Text"] == text :
                data.loc[i, "Text"] = "Esterilización"
                count += 1

            elif data.loc[i, "Text"] == "LLenado" :
                data.loc[i, "Text"] = "Llenado"

        text_List.clear()
        for i in range(1, rows, 1):
            if(text_List.__contains__(data.loc[i, "Text"])== False):
                text_List.append(data.loc[i, "Text"])

        df = pd.DataFrame(data=text_List)
        #df.to_csv("D:\\CGN\\projects\\AutoclaveFailDeteccion\\data\\Datos\\phases_names.csv", index=True, header=True)
        
        data.to_csv(path, columns=["EntityId", "ExecutionId", "Time", "Text"], index=False)


    def remove_incorrect_time(self):
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
        
        self._logger.info(f"Executions.Count = {len(executions)} \nCorrect_executions.count = {len(correct_exc)}")


    def split_sequences(self):
        """Split the Executions to the correspondent sequence,
        generate N folders named as the correspondent sequence
        and inside generate a csv that contains the executions of that sequence 
        Also when save the (sequence_name)_phases.csv create a new column End Date that contains the End Time of the phase
        """
        path_phases_not_contained_in_any_sequence = os.path.join(self._base_directory, self._data_analysis, 'phases_not_contained_in_any_sequence.csv')
        if(os.path.exists(path_phases_not_contained_in_any_sequence)):
            self._logger.info("The executions are allready splited by sequence, nothing to do in execution_analyzer.split_sequences()")
            return
        
        executions_path = os.path.join(self._base_directory, self._executions_to_filter)
        phases_path = os.path.join(self._base_directory, self._phases_directory)
        
        executions = pd.read_csv(executions_path)
        phases = pd.read_csv(phases_path)
        
        self._sequences_names = executions.loc[:, 'SequenceName'].drop_duplicates() #obtain the names of each sequence in the executions
                
        phases_ids_contained = list()
        
        for sequence in self._sequences_names: #itetate on each sequence to assing the executions that belong to each one
            
            folder_to_save = os.path.join(self._base_directory, self._data_analysis, sequence)
            if(os.path.exists(folder_to_save) == False):
                 os.makedirs(folder_to_save)
            
            path_to_save = os.path.join(folder_to_save, f"{sequence}_phases.csv") #path where will be saved the phases that belong to the current sequence
            
            temp_ids = executions.apply(lambda row: row['ExecutionId'] if row['SequenceName'] == sequence else np.nan
                                        , axis=1) #iterate over rows to assing the Executions IDs that belong to that sequence
            
            temp_ids.dropna(inplace=True) # delete the NaN and keep the IDs, this IDs will be used to find the Phases of this sequence
            temp_ids = temp_ids.to_numpy()
                    
            for id in temp_ids:
                 phases_ids_contained.append(id) #saving the Ids of the phases that belong to the present sequence to the list where are saved 
                                                  #the Ids of the phases in each sequence
                        
            temp_phases = phases[phases['ExecutionId'].isin(temp_ids)]
            
            #end_time = temp_phases_copy['Time']
            end_time_column = temp_phases.iloc[1:temp_phases.__len__(), 2] #obtain from row 1 to end all the times
                                                                    #this dataframe is to obtain the start time of the next phase that will be the end time 
                                                                    #of the actual phase if they belong to the same Execution ID, else the EndTime of the Execution is
                                                                    #the End time of the phase
            end_time_column = end_time_column.reset_index()
            end_time_column = end_time_column['Time'].to_numpy()
            last_execution_id = temp_ids[temp_ids.__len__() - 1] #the id of the last execution of the sequence
            last_execution_end_time = executions.loc[executions['ExecutionId'] == last_execution_id, 'EndTime'] # the end time of the last execution
            
            
            end_time_column = np.append(end_time_column, last_execution_end_time)
            
            temp_phases.insert(3, column="EndTime", value=end_time_column)
            
            '''
            b = executions.loc[executions['ExecutionId'] == 1, 'EndTime'].item()
            print(b)
            a = temp_phases.loc[0, 'EndTime']
            pas = a > b
            
            date_str = temp_phases.loc[0, 'EndTime']
                                                                                                 
            print(datetime.strptime(date_str, self._date_time_format))
            '''
            
            temp_phases['EndTime'] = temp_phases.apply(lambda row: row['EndTime'] if datetime.strptime(row['EndTime'], self._date_time_format) < datetime.strptime(executions.loc[executions['ExecutionId'] == row['ExecutionId'], 'EndTime'].item(), self._date_time_format) 
                                                                                 else executions.loc[executions['ExecutionId'] == row['ExecutionId'], 'EndTime'].item()
                                                                                 , axis=1) #adding the correct end time column 
                                        
            temp_phases.to_csv(path_to_save, index=False, header=['EntityId', 'ExecutionId', 'StartTime', 'EndTime', 'Text']) #saving the phases that belong to each sequence
            
        
        index_phases_containeds = phases['ExecutionId'].isin(phases_ids_contained)
        index_phases_not_containeds = index_phases_containeds.apply(lambda row: not row)
        phases_not_contained = phases[index_phases_not_containeds]
        
        phases_not_contained.to_csv(path_phases_not_contained_in_any_sequence, index=False, header=True)
        
    
    def filter_samples_by_phases(self, phase_conf):
        """

        Args:
            phase_conf (obj): the conf of the phase
        """
        samples = pd.read_csv(os.path(self._base_directory, self._samples_directory))
        phases = pd.read_csv(os.path.join(self._base_directory, self._data_analysis, self._sequence_directory, self._phases_by_sequence_directory))
        
        
            
        
        
