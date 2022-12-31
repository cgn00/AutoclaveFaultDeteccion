import pandas as pd
import numpy as np
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import DBSCAN
from collections import Counter
from configuration.phase_conf import sequence_config
from dtaidistance import dtw

class executions_analyzer:
    
    """This class is to explore the executions, agrupate all the executions by a given phase...
    """
    
    #construct:
    
    def __init__(self, path):
        self._base_directory = path 
        self._duration_directory = 'Durations\\'
        self._variables_directory = "Variables\\"
        self._data_directory = "Datas\\"
        self._samples_directory = "Samples For Executions Cluster\\"
        self._samples_csv_directory = 'Datos\\samples.csv'
        self._phases_with_mistakes_in_name_directory = 'Datos\\phases.csv'    # the phases: Esterilizacion an Llenado appear with difirents mistakes in the name
        self._phases_directory = "Datos\\phases_to_analysis.csv"
        self._executions_directory = "Datos\\clean_executions.csv"
        self._executions_to_filter_directory = "Datos\\executions.csv"
        self._data_analysis_directory = 'Data_Analysis'
        self._distances_dtw_directory = 'Distances_DTW\\'
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
        
        path = os.path.join(self._base_directory, self._phases_with_mistakes_in_name_directory)
        
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


    def remove_executions_with_incorrect_time(self):
        """
        This function remove executions with incorrect Start or End Time.
        The Start Time can't happen after the End Time.
        Generate at ...Datos\\ a "clean_executions.csv" file with the corrects executions
        It's the equivalent CleanExecutionsCSV function of the analysis class in C#
        """
        
        path_to_read = os.path.join(self._base_directory, self._executions_to_filter_directory)
        path_to_save = os.path.join(self._base_directory, self._executions_directory)
        
        if(os.path.exists(path_to_save)):
            self._logger.info("Executions with incorrect Times were already removed, there isn't nothing to remove ")
            
            return
        
        executions = pd.read_csv(path_to_read, sep=',')
        executions.sort_values(by=['StartTime'], inplace=True) # sort the dataframe by StartTime criterion
        
        correct_exc = pd.DataFrame(columns=executions.columns)
        
        """
        if(executions.loc[0, 'StartTime'] < executions.loc[0, 'EndTime']):
            correct_exc.loc[len(correct_exc)] = executions.loc[0, :]
        """
        
        #rows_count = executions.shape[0]
        
        correct_exc = executions[executions['StartTime'] < executions['EndTime']]
        
        '''
        for i in range(0, rows_count, 1):
            #if(executions.loc[i, 'StartTime'] >= correct_exc.loc[len(correct_exc) - 1, 'EndTime'] and executions.loc[i, 'StartTime'] < executions.loc[i,'EndTime']):
            if(executions.loc[i, 'StartTime'] < executions.loc[i, 'EndTime']):
                #the atual execution has a correct start and end times
                correct_exc.loc[len(correct_exc)] = executions.loc[i, :] #add all the columns of the row
        
        '''
        
        correct_exc.to_csv(path_to_save, header=True, index=False)
        
        self._logger.info(f"Executions.Count = {len(executions)} \nCorrect_executions.count = {len(correct_exc)}")


    def split_sequences(self):
        """Split the Executions to the correspondent sequence,
        generate N folders named as the correspondent sequence
        and inside generate a csv that contains the executions of that sequence 
        Also when save the (sequence_name)_phases.csv create a new column End Date that contains the End Time of the phase.
        Also initialize the self._sequences_names
        """
               
        executions_path = os.path.join(self._base_directory, self._executions_directory)
        phases_path = os.path.join(self._base_directory, self._phases_directory)
        
        executions = pd.read_csv(executions_path)
        phases = pd.read_csv(phases_path)
        
        #phases = phases.sort_values(by=['StartTime']) #sort the phases by Start time criteria
        
        self._sequences_names = executions.loc[:, 'SequenceName'].drop_duplicates() #obtain the names of each sequence in the executions
                
        path_phases_not_contained_in_any_sequence = os.path.join(self._base_directory, self._data_analysis_directory, 'phases_not_contained_in_any_sequence.csv')
        if(os.path.exists(path_phases_not_contained_in_any_sequence)):
            self._logger.info("The executions are allready splited by sequence, nothing to do in execution_analyzer.split_sequences()")
            return 
                
        phases_ids_contained = list()
        
        for sequence in self._sequences_names: #iterate on each sequence to assing the executions that belong to each one
            
            folder_to_save = os.path.join(self._base_directory, self._data_analysis_directory, sequence)
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
        This function find all the samples of the phase_conf recived and Will create phase_name_samples.csv file
        where are saved all the samples of the phase_conf recived. 
        Also generate a phase_name_data.csv file that contains one column for the respective variableId with the time serie of the samples
        Only saves the samples wichs 'SampleId' is containned in self.variables_ids.
        The samples saved are sorted by 'Time'
        Args:
            phase_conf (obj: phase_config from phase_conf.py module): here are the configurations of the phase
        """
        
        #ask if the directory exists to return then
        path_to_save = os.path.join(self._sequence_directory, phase_conf._name)
        if(os.path.exists(os.path.join(path_to_save, phase_conf._name +  '_data.csv')) == True):
            self._logger.info(f"The samples of the phase {phase_conf._name} are allready splited by phase (the file {phase_conf._name}_data.csv allready exists), nothing to do in execution_analyzer.filter_samples_by_phases(phase_conf)")
            return
        
        if(os.path.exists(path_to_save) == False):
            os.makedirs(path_to_save)
        
        samples_path = os.path.join(self._base_directory, self._samples_csv_directory)
        samples = pd.read_csv(samples_path)
        
        samples = samples[samples['VariableId'].isin(self.variables_ids)]#remove the samples with ids that arn't at self.variables_ids, the 6 temp and 1 presuere
        
        phases = pd.read_csv(os.path.join(self._base_directory, self._data_analysis_directory, self._sequence_directory, self._phases_by_sequence_directory))
        
        phases['StartTime'] = pd.to_datetime(phases['StartTime'], format=self._date_time_format)
        phases['EndTime'] = pd.to_datetime(phases['EndTime'], format=self._date_time_format)
        
        samples['Time'] = pd.to_datetime(samples['Time'],format=self._date_time_format)
        
        phase_name = phase_conf._name #the name of the actual phase 
        
        all_exec_of_one_phase = phases.loc[phases['Text'] == phase_name, 'EntityId':'EndTime'] #crate a dataframe with Execution Id, EntityId, Start and End Time columns of the phase recived
        all_exec_of_one_phase.sort_values(by=['StartTime'], inplace=True)# sort ascending the phases by time
        
        sorted_samples = samples.sort_values(by=['Time']) # sort ascending the samples by time
        
        start = all_exec_of_one_phase['StartTime'].iloc[0] 
        end = all_exec_of_one_phase['EndTime'].iloc[all_exec_of_one_phase.__len__()-1]
        
        correct_samples = sorted_samples[(sorted_samples['Time'] >= start) & (sorted_samples['Time'] <= end)] #these are all the samples in the range of the actual phase start and end time of the first and last execution,
                            #but aren't labeled by ExecutionId or EntityId yet, there are also samples that doesn't belong to the phase that will be removed later
        
        correct_samples.loc[:, 'EntityId'] = np.nan #this column is the unique id of the phase execution
        correct_samples.loc[:, 'ExecutionId'] = np.nan #this column is the id of the entire execution (an execution contains diferents EntityId, one for each phase executed in the execution)
                
        
        headers = ['ExecutionId', 'EntityId', 'Time', 'SampleId']
        for id in self.variables_ids:
            headers.append(str('Variable_Id_' + str(id))) #each variable Id from self.variables_ids will represent a column at the dataframe
                
        data = pd.DataFrame(columns=headers)
        data['Time'] = pd.to_datetime(data['Time'],format=self._date_time_format)
        
        data['Time'] = correct_samples['Time'].unique() #assign the time of each sample
        
                
        for index, phase_row in all_exec_of_one_phase.iterrows(): #iterate over each phase execution to assign the ExecutionId and EntityId to each sample
            boolean = (correct_samples['Time'] >= phase_row.loc['StartTime']) & (correct_samples['Time'] < phase_row.loc['EndTime']) #return a true and false column with the samples of the actual phase execution
            
            correct_samples.loc[boolean, 'EntityId'] = phase_row['EntityId'] #assign the EntityId and the ExecutionId to the samples of the actual phase execution
            correct_samples.loc[boolean, 'ExecutionId'] = phase_row['ExecutionId']
            
            
            time_serie = correct_samples[boolean] #this is the time serie with all the samples of all the variables of the actual phase execution
            
            samples_times = time_serie['Time'].unique() #these are the time when were taked the measurement of each variable
            
            sample_id = 0
            
            for time in samples_times: #iterate over each unique time of the time serie
                measure_of_one_time = time_serie[time_serie['Time'] == time] #obtain the 7 samples(one sample for each variable) that belong to the actual time
                
                sample_id += 1
                
                location = data['Time'] == time
                
                #data[location].loc['Time'] = time                                                            
                data.loc[location, 'ExecutionId'] = phase_row.loc['ExecutionId']
                data.loc[location, 'EntityId'] = phase_row.loc['EntityId']
                data.loc[location, 'SampleId'] = sample_id # assign 'SampleId'
                
                               
                for sample in measure_of_one_time.itertuples(index=False): #iterate over each variable of the samples that were measured in the actual time
                    variableId_position = str('Variable_Id_' + str(sample[2])) #the varible id of the sample will say to wich column at the data(dataframe) assign the value of the sample
                    data.loc[location, variableId_position] = sample[1] # the data at the column that correspond with the variableId of the sample will be assigned the value of this measurement
            
            phases.loc[phases['EntityId'] == phase_row.loc['EntityId'], 'SampleCount'] = sample_id #save the number of samples that the executions of the phase(time serie) has
            
        
        phases.to_csv(os.path.join(self._base_directory, self._data_analysis_directory, self._sequence_directory, self._phases_by_sequence_directory) , index=False, header=True)
        
        correct_samples.dropna(inplace=True) #remove the samples that not belong to the phase
        
        correct_samples.to_csv(os.path.join(path_to_save , phase_conf._name +  '_samples.csv'), index=False, header=True)
        
        data.dropna(inplace=True) #remove the samples that not belong to the phase
        
        path = os.path.join(self._sequence_directory, phase_conf._name)
        file_to_save = os.path.join(path, phase_conf._name +  '_data.csv')
        data.to_csv(file_to_save, header=True, index=False)
            
            
    def save_data_csv(self, phase_conf):
        """_summary_
        
        Generate a phase_name_data.csv file that contains one column for the respective variableId
        
        The function: filter_samples_by_phases(self, phase_conf) do the task of this function more efficient and fast

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
        

    def remove_incorrect_time_series(self, phase_conf, sequence_name):
        """Remove from the analysis the time series that do not satisfy the phase_conf minimun number of samples and time criteria.
        Also remove the time series that don't have the linnear relation between time and number of samples(number of samples = 2 samples per minute, y=2t)

        Args:
            phase_conf (obj: phase_config from phase_conf.py module): here are the configurations of the phase
        """
        
        data_path = os.path.join(self._sequence_directory, phase_conf._name, phase_conf._name +  '_data.csv')
        
        phases = pd.read_csv(os.path.join(self._base_directory, self._data_analysis_directory, self._sequence_directory, self._phases_by_sequence_directory))
        phases['StartTime'] = pd.to_datetime(phases['StartTime'], format=self._date_time_format)
        phases['EndTime'] = pd.to_datetime(phases['EndTime'], format=self._date_time_format)
        
        phases = phases[phases['Text'] == phase_conf._name] #select only the phases of the type of phase_conf
        
        #duration = phases['EndTime'] - phases['StartTime']
                      
        boolean = (phases['SampleCount'] >= phase_conf._samples_count) #remove the time series that not satisfy the minimun number of samples of the phase
        
        corrects_phases = phases.loc[boolean]
        bad_phases = phases.loc[~boolean]
        
        time_serie_duration_minutes = corrects_phases.apply(lambda row: (row['EndTime'] - row['StartTime']).total_seconds() /60
                                                   , axis=1) #the duration in minutes of each execution of the phase(the time serie of the samples)
        
        time_serie_number_of_samples = corrects_phases['SampleCount'] #the number of samples that the time serie has
        
        start_time = 0
        end_time = int(time_serie_duration_minutes.max().round()) #start and end time of the plot
        
        x = np.arange(start=start_time, stop=end_time+1) #the stop time isn't include in np.arange()
        y = 2*time_serie_duration_minutes.to_numpy() # y: is the expected number of samples evalueted in 2*(duration in minutes)
        
        ones = np.ones(x.shape)
        
        error = np.power((y - time_serie_number_of_samples.to_numpy()), 2) #cuadratic error between the expected and the real number of samples of the time serie
        mean = error.mean()
        std = error.std()
        
        mean = ones * mean
        std = ones * std
        
        bad_phases = bad_phases.append(corrects_phases[~(error < (error.mean() + error.std())) & (error > (error.mean() - error.std()))])#obtain the executions of the phases that are outside the mean +- std of the error 
                                    #(the ~ negate the true-false serie and give the opositive)  
        bad_executions_ids = bad_phases['ExecutionId']
        
        corrects_phases = corrects_phases[(error < (error.mean() + error.std())) & (error > (error.mean() - error.std()))] #remove the executions of the phases that are outside the mean +- std of the error       
        good_executions_ids = corrects_phases['ExecutionId']
        
        data = pd.read_csv(data_path) #read the data to remove the time serie with bad executions ids
        
        index_to_drop = data[data['ExecutionId'].isin(bad_executions_ids)].index
        
        good_data = data.drop(index=index_to_drop) #this is the data of the time serie with the good executions ids(removed the bad executions ids)
        
        good_data.to_csv(data_path, index=False, header=True) #saving the good data as data.csv
        
        #---VISUALIZATION---
        data_durations = pd.concat([time_serie_duration_minutes, time_serie_number_of_samples], axis=1,)
        data_durations.columns = ['time', 'num_samp']
        
        clusters = DBSCAN(eps=3.5, min_samples=5).fit(data_durations)
        
        '''
        plt.subplot(2, 1, 1)
        p = sns.scatterplot(data=data_durations, x="time", y="num_samp" ,hue=clusters.labels_, legend="full", palette="deep")
        #sns.move_legend(p, loc = "upper right", bbox_to_anchor = (1.17, 1.12), title = 'Clusters')
        plt.title("Sequence: " + sequence_name + "\nPhase: " + phase_conf._name + "\n" + str(Counter(clusters.labels_)))
        #plt.text(0, 200, str(Counter(clusters.labels_)))
        
        plt.plot(time_serie_duration_minutes, y)
        
        plt.subplot(2, 1, 2)
        plt.scatter(time_serie_duration_minutes, error, marker='.')
        plt.plot(x, mean)
        plt.plot(x, mean + std, color='red', linestyle='dashed')
        plt.plot(x, mean - std, color='red', linestyle='dashed')
        plt.show()
        plt.close()
        '''
        
    def calculate_dtw_metrics(self, phase_conf):
        """For each variable(in self.variables_ids) calculate the distances between the time series on each execution of the phase received.
        These distances will be used to determinate clusters with DBSCAN to clasify good and fail executions of the phase received

        Args:
            phase_conf (obj: phase_config from phase_conf.py module): here are the configurations of the phase        
        """

        data_path = os.path.join(self._sequence_directory, phase_conf._name, phase_conf._name +  '_data.csv')
        data = pd.read_csv(data_path) #load the data.csv where are the time series of each variable
        
        executions_ids = data['ExecutionId'].unique()
        
        for variab_id in self.variables_ids: #iterate over each variableId (7..13)
            
            path_to_save = os.path.join(self._sequence_directory, phase_conf._name, self._distances_dtw_directory)
            if(os.path.exists(path_to_save) == False):    
                os.makedirs(path_to_save)
            file_path = os.path.join(path_to_save, f'distances_variable_{variab_id}.csv')
            
            if(os.path.exists(file_path)):
                self._logger.info(f"The DTW distances {file_path} of the phase {phase_conf._name} are allready, nothing to calculate with the variableId: {variab_id}")
                continue #jump to the next variableId
            
            time_series_list = []
                              
            for exe_id in executions_ids: #iterate over each execution to obtain the time serie of each execution
                temp_time_serie = data.loc[data['ExecutionId'] == exe_id, f'Variable_Id_{variab_id}'].to_numpy() #obtain the time serie of the variable in the iteration

                time_series_list.append(temp_time_serie) #add the time serie to the list of this execution
            
            time_series_one_variable = np.array(time_series_list) #convert the list of times series into a np.array
            
            length = time_series_one_variable.__len__() #the number of time series (# of executions of the phase)
            
            distance = np.zeros(length)
            
            for j in range(length): #iterate over each time serie
                
                temp_distance = np.zeros(length) #temporal distances between one time serie and the other, to calculate the median of them
                
                for k in range(length): #iterate over each time serie
                    
                    #x = time_series_one_variable[j] #the time serie that will be compare to the others and keep de median of the distances
                    
                    #y = time_series_one_variable[k] #the time serie to compare to (x) 
                    
                    temp_distance[k] = dtw.distance_fast(time_series_one_variable[j], time_series_one_variable[k])
                    
                
                distance[j] = np.median(temp_distance) #keep the median of the distances from the time serie [j] with respect to the others
                
            distance_df = pd.DataFrame(distance)
            
            distance_df.to_csv(file_path, index=False, header=False)
                    
                  
                    
         