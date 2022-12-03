close all, clear all, clc

<<<<<<<< HEAD:Proces_in_matlab/analysis_dtw.m
location = 'D:\CGN\projects\AutoclaveFailDeteccion\data\Samples Sorts by Phases\';
========
fileID = fopen('baseDirectory.txt','r');
base_directory = fscanf(fileID,'%c')
fclose(fileID);

sequence_name = "Prueba de estanco";
location = base_directory + sequence_name + "\Samples Sorts by Phases\";
%location = 'D:\CGN\projects\AutoclaveFailDeteccion\data\Samples Sorts by Phases\';
>>>>>>>> origin/preprocessing_without_mlnet:dtw_and_dbscan_in_matlab/analysis_dtw.m
phase = 'Presurización';
file = '\Variables\variable';
extension = '.csv';


<<<<<<<< HEAD:Proces_in_matlab/analysis_dtw.m
for i=1:7
========
for i=7:13
>>>>>>>> origin/preprocessing_without_mlnet:dtw_and_dbscan_in_matlab/analysis_dtw.m
    
    variable = readmatrix(char(string(location)+string(phase)+string(file)+string(i)+string(extension)));      
    variable = normalize(variable,'range');
    p = variable.';

    %distance = [];
    distance = zeros(length(p(1,:)),1);
    
    % Calcular distancias

    for j = 1:(length(p(1,:)))

        temp_distance = zeros(1, length(p(1,:)));
        
        for k = 1:(length(p(1,:)))

            % Eliminar valores NaN 
            x_temp(1,:) = (p(:,j));
            TF = ~isnan(x_temp(1,:));
            x(1,:) = x_temp(TF);

            % Eliminar valores NaN 
            y_temp(1,:) = (p(:,k));
            TF = ~isnan(y_temp(1,:));
            y(1,:) = y_temp(TF);

            % Calcular distancias entre señales
            temp = dtw(x,y);

            temp_distance(k) = temp;

            x = [];
            x_temp = [];
            y = [];
            y_temp = [];

        end

        % Quedarse con la mediana
        distance(j,1) = median(temp_distance);

    end

    % Save distances
<<<<<<<< HEAD:Proces_in_matlab/analysis_dtw.m
    writematrix(distance,char(string(location)+string(phase)+'\distances_variable'+string(i)+string(extension)));
========
    path = char(string(location)+string(phase))
    %mkdir path matlab
    writematrix(distance,char(string(location)+string(phase)+ '\distances_variable'+string(i)+string(extension)));
>>>>>>>> origin/preprocessing_without_mlnet:dtw_and_dbscan_in_matlab/analysis_dtw.m

    
end



