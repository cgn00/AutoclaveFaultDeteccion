import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
from collections import Counter

def clusterByTimeAndSamples():
    base_directory_txt_path = os.getcwd().removesuffix('Proces_in_python') + "baseDirectory.txt"
    base_directory = pd.read_csv(base_directory_txt_path, header=None).iloc[0].values[0] #reading the base deirectory where is allocated the Data
    #base_directory = pd.read_csv()
    sequences = {"Esterilización": 0, "Prueba de estanco": 0}
    
    for seq in sequences:
        path = base_directory + 'Data_Analysis\\' + seq + "\\"  
        df = pd.read_csv(path  + seq + "_phases.csv", header=None) #reading the csv that contains the phases that belong to the sequence
        sequences[seq] = df[3].unique() #asign the phases names to each sequence
        
        for phase in sequences[seq]:
            IDs_to_save_path = path + "Samples Sorts by Phases\\" + phase + "\\CorrectExecutionsByTimeAndSamples.csv"
            
            ids_to_read_path = path + "Samples Sorts by Phases\\" + phase + "\\executions_with_samples.csv"
            ids = pd.read_csv(ids_to_read_path, header=None)
            
            durations_path = path + "\\Samples Sorts by Phases\\" + phase + "\\Durations\\" #location where are saved the Durations by time and number of samples
            durations_by_time = pd.read_csv(durations_path + "durations.csv", header=None)
            durations_by_samples = pd.read_csv(durations_path + "NumberOfSamplesBeforeClean.csv", header=None)
            
            data_durations = pd.concat([durations_by_time, durations_by_samples], axis=1,)
            data_durations.columns = ['time', 'num_samp']
            #print(data_durations)
            #clustering = pd.DataFrame(DBSCAN(eps=3, min_samples=2).fit_predict(data_durations[['time', 'num_samp']]))
            clusters = DBSCAN(eps=3.5, min_samples=5).fit(data_durations)
            
            ids_with_clusters = ids[clusters.labels_ != -1]
            if(os.path.isfile(IDs_to_save_path)): #if the file exist is deleted and rewrited
                os.remove(IDs_to_save_path)
                
            ids_with_clusters.to_csv(IDs_to_save_path, index=False, header=False)
            
            #print(clusters.labels_)
            print(Counter(clusters.labels_))
            
            p = sns.scatterplot(data=data_durations, x="time", y="num_samp" ,hue=clusters.labels_, legend="full", palette="deep")
            #sns.move_legend(p, loc = "upper right", bbox_to_anchor = (1.17, 1.12), title = 'Clusters')
            plt.title("Sequence: " + seq + "\nPhase: " + phase + "\n" + str(Counter(clusters.labels_)))
            #plt.text(0, 200, str(Counter(clusters.labels_)))
            plt.show()
            plt.close()
    
    
if __name__ == "__main__":
    clusterByTimeAndSamples()