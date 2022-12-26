import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import json
from configuration.phase_conf import sequence_config

class executions_analyzer:
    
    """This class is to explore the executions, agrupate all the executions by a given phase...
    """
    
    #construct:
    
    def __init__(self, path):
        self._base_directory = path 
        self._duration_directory = '\\Durations\\'
        self._variables_directory = "\\Variables\\"
        self._data_directory = "\\Datas\\"
        self._samples_directory = "Samples For Executions Cluster\\"
        self._samples_csv_path = 'Datos\\samples.csv'
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
        
        self.variables_ids = [7, 8, 9, 10, 11, 12, 13] #these are the ids of the 7 variables(6 temperatures and 1 presure)
                
        self._sequences_names = list()
        
        self.load_phase_conf_json() #initialize the self._sequences_config obj
        
        self.logger_config() #logger config
          
      
    #config functions that will be used in the construct: __init__():
    
    def load_phase_conf_json(self):
        """
        Initialize the self._sequences_config obj
        """
        route = os.path.join(self._base_directory, self._phase_configuration_directory)
        #self._phase_config = json.load(open(route)) #load the Phase Configuration saved as .json
        seqns_dic = json.load(open(route, encoding="utf-8")) # encoding="utf-8" is the configuration to read characters in spanish(ó,á,é,ú,í,ñ..) from a file 
        self._sequences_config = []
        
        for seq in seqns_dic['sequences_config']:
            self._sequences_config.append(sequence_config(seq)) #initialize the construct of each sequence config and add it to a list of sequences config
          
            
    def logger_config(self):
        """
        This function configure the logger
        """
        
        self._logger = logging.getLogger('AnalyzerLogger')
        self._logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler('logger.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.info("--------------------New Execution of the code-------------------")
        
        
    #principals functions:            
            
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
        or The execution (N-1) can't end after the execution (N) start
        It's the equivalent CleanExecutionsCSV function of the analysis class in C#
        """
        
        path_to_read = os.path.join(self._base_directory, self._executions_to_filter)
        path_to_save = os.path.join(self._base_directory, self._executions_directory)
        
        if(os.path.exists(path_to_save)):
            self._logger.info("Executions with incorrect Times were already removed, there isn't nothing to remove ")
            
            return
        
        executions = pd.read_csv(path_to_read, sep=',')
        executions.sort_values(by=['StartTime'], inplace=True) # sort the dataframe by StartTime criterion
        
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
        Also when save the (sequence_name)_phases.csv create a new column End Date that contains the End Time of the phase.
        Also initialize the self._sequences_names
        """
               
        executions_path = os.path.join(self._base_directory, self._executions_to_filter)
        phases_path = os.path.join(self._base_directory, self._phases_directory)
        
        executions = pd.read_csv(executions_path)
        phases = pd.read_csv(phases_path)
        
        #phases = phases.sort_values(by=['StartTime']) #sort the phases by Start time criteria
        
        self._sequences_names = executions.loc[:, 'SequenceName'].drop_duplicates() #obtain the names of each sequence in the executions
                
        path_phases_not_contained_in_any_sequence = os.path.join(self._base_directory, self._data_analysis, 'phases_not_contained_in_any_sequence.csv')
        if(os.path.exists(path_phases_not_contained_in_any_sequence)):
            self._logger.info("The executions are allready splited by sequence, nothing to do in execution_analyzer.split_sequences()")
            return 
                
        phases_ids_contained = list()
        
        for sequence in self._sequences_names: #iterate on each sequence to assing the executions that belong to each one
            
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
                                        
            temp_phases.sort_values(by=['Time'], inplace=True) #sort the phases by StartTime criterion
            temp_phases.to_csv(path_to_save, index=False, header=['EntityId', 'ExecutionId', 'StartTime', 'EndTime', 'Text']) #saving the phases that belong to each sequence
            
        
        index_phases_containeds = phases['ExecutionId'].isin(phases_ids_contained)
        index_phases_not_containeds = index_phases_containeds.apply(lambda row: not row)
        phases_not_contained = phases[index_phases_not_containeds]
        
        phases_not_contained.to_csv(path_phases_not_contained_in_any_sequence, index=False, header=True)
        
    
    def filter_samples_by_phases(self, phase_conf):
        """
        This function find all the samples of the phase_conf recived and Will create phase_name_samples.csv files
        where are saved all the samples of the phase_conf recived. Only saves the samples wichs 'SampleId' is containned in self.variables_ids.
        The samples saved are sorted by 'Time'
        Args:
            phase_conf (obj: phase_config from phase_conf.py module): here are the configurations of the phase
        """
        
        #ask if the directory exists to return then
        path_to_save = os.path.join(self._sequence_directory, phase_conf._name)
        if(os.path.exists(os.path.join(path_to_save, phase_conf._name +  '_samples.csv')) == True):
            self._logger.info(f"The samples of the phase {phase_conf._name} are allready splited by phase, nothing to do in execution_analyzer.filter_samples_by_phases(phase_conf)")
            return
        
        if(os.path.exists(path_to_save) == False):
            os.makedirs(path_to_save)
        
        samples_path = os.path.join(self._base_directory, self._samples_csv_path)
        samples = pd.read_csv(samples_path)
        
        samples = samples[samples['VariableId'].isin(self.variables_ids)]#remove the samples with ids that arn't at self.variables_ids, the 6 temp and 1 presuere
        
        phases = pd.read_csv(os.path.join(self._base_directory, self._data_analysis, self._sequence_directory, self._phases_by_sequence_directory))
        
        phases['StartTime'] = pd.to_datetime(phases['StartTime'], format=self._date_time_format)
        phases['EndTime'] = pd.to_datetime(phases['EndTime'], format=self._date_time_format)
        
        samples['Time'] = pd.to_datetime(samples['Time'],format=self._date_time_format)
        
        phase_name = phase_conf._name #the name of the actual phase 
        
        start_end_times_of_phase = phases.loc[phases['Text'] == phase_name, 'EntityId':'EndTime'] #crate a dataframe with Execution Id, Start and End Time columns of the actual phase
        start_end_times_of_phase.sort_values(by=['StartTime'], inplace=True)
        
        sorted_samples = samples.sort_values(by=['Time']) # sort ascending the samples by time
        
        start = start_end_times_of_phase['StartTime'].iloc[0]
        end = start_end_times_of_phase['EndTime'].iloc[start_end_times_of_phase.__len__()-1]
        
        correct_samples = sorted_samples[(sorted_samples['Time'] >= start) & (sorted_samples['Time'] <= end)] #these are the samples of the actual phase, but aren't labeled by ExecutionId or EntityId yet
        #correct_samples['EntityId'] = np.nan
        correct_samples.loc[:, 'EntityId'] = np.nan #this column is the unique id of the phase execution
        correct_samples.loc[:, 'ExecutionId'] = np.nan #this column is the id of the entire execution (an execution contains diferents EntityId, one for each phase executed in the execution)
        
        for index, phase_row in start_end_times_of_phase.iterrows(): #iterate over the phase executions to assign the ExecutionId and EntityId to each sample
            boolean = (sorted_samples['Time'] >= phase_row.loc['StartTime']) & (sorted_samples['Time'] <= phase_row.loc['EndTime']) #return a true and false column with the samples of the actual phase execution
            
            correct_samples.loc[boolean, 'EntityId'] = phase_row['EntityId'] #assign the EntityId and the ExecutionId to the samples of the actual phase execution
            correct_samples.loc[boolean, 'ExecutionId'] = phase_row['ExecutionId']
            
        correct_samples.dropna(inplace=True)
        

        correct_samples.to_csv(os.path.join(path_to_save , phase_conf._name +  '_samples.csv'), index=False, header=True)
            
            
    def save_data_csv(self, phase_conf):
        """_summary_
        
        Generate a phase_name_data.csv file that contains one column for the respective variableId

        Args:
            phase_conf (obj: phase_config from phase_conf.py module): here are the configurations of the phase
        """
        
        #ask if the directory exists to return then
        path = os.path.join(self._sequence_directory, phase_conf._name)
        file_to_read = os.path.join(path, phase_conf._name +  '_samples.csv')
        file_to_save = os.path.join(path, phase_conf._name +  '_data.csv')
         
        if(os.path.exists(file_to_save)):
            self._logger.info(f"The  {file_to_save} allready exists, nothing to do in execution_analyzer.save_data_csv(phase_conf)")
            return
        
        samples = pd.read_csv(file_to_read)
        #samples['Time'] = pd.to_datetime(samples['Time'], format=self._date_time_format) #declarate the column 'Time' as DateTime type
        
        headers = ['ExecutionId', 'EntityId', 'Time', 'SampleId']
        for id in self.variables_ids:
            headers.append(str('Variable_Id_' + str(id))) #each variable Id from self.variables_ids will represent a column at the dataframe
                
        data = pd.DataFrame(columns=headers)
        
        data['Time'] = samples['Time'].unique() #assign the time of each sample
        
        sample_id = 0
        last_entityId = samples.iloc[0, 4] #this is the Entity Id of the first sample first 
        
        for index, data_row in data.iterrows(): #iterate over each row of the dataframe to obtain the time of the sample
            samples_searched_by_time = samples[samples['Time'] == data_row['Time']] #obtain the 7 samples(one sample for each variable) that belong to the actual time
            
            actual_entityId = samples_searched_by_time.iloc[0, 4] #assign 'EntityId'
            
            if(last_entityId != actual_entityId): #if the actual sample belong to a new EntityId then reset the sample id
                sample_id = 1 #restart the counter
            else: #if the actual sample doesn't belong to a new EntityId then increment the sample id
                sample_id += 1
            
            data_row['ExecutionId'] = samples_searched_by_time.iloc[0, 5] #assign 'ExecutionId'
            data_row['EntityId'] = samples_searched_by_time.iloc[0, 4]  #assign 'EntityId'
            data_row['SampleId'] = sample_id # assign 'SampleId'
            
            for sample in samples_searched_by_time.itertuples(index=False): #iterate over each variable of the samples that were measured in the actual time
                variableId_position = str('Variable_Id_' + str(sample[2])) #the varible id of the sample will say to wich column at the data(dataframe) assign the value of the sample
                data_row[variableId_position] = sample[1] # the data_row at the column that correspond with the variableId of the sample will be assigned the value of this measurement
                   
            
            last_entityId = actual_entityId #in the next iteration the last_entityId is the actual
                       
             
        data.to_csv(file_to_save, header=True, index=False)
        

       