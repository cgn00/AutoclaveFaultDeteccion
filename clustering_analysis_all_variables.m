close all, clear all, clc

% Load data
times = readmatrix('durations_good_executions.csv');

% Load distances
distances1 = readmatrix('distances_variable1.csv');
distances2 = readmatrix('distances_variable2.csv');
distances3 = readmatrix('distances_variable3.csv');
distances4 = readmatrix('distances_variable4.csv');
distances5 = readmatrix('distances_variable5.csv');
distances6 = readmatrix('distances_variable6.csv');
distances7 = readmatrix('distances_variable7.csv');
distances8 = readmatrix('distances_variable8.csv');
distances9 = readmatrix('distances_variable9.csv');
distances10 = readmatrix('distances_variable10.csv');
distances11 = readmatrix('distances_variable11.csv');
distances12 = readmatrix('distances_variable12.csv');
distances13 = readmatrix('distances_variable13.csv');
distances14 = readmatrix('distances_variable14.csv');
distances15 = readmatrix('distances_variable15.csv');
distances16 = readmatrix('distances_variable16.csv');
distances17 = readmatrix('distances_variable17.csv');


distances1 = distances1.';
distances1 = normalize(distances1,'zscore');
distances1 = distances1.';

distances2 = distances2.';
distances2 = normalize(distances2,'zscore');
distances2 = distances2.';

distances3 = distances3.';
distances3 = normalize(distances3,'zscore');
distances3 = distances3.';

distances4 = distances4.';
distances4 = normalize(distances4,'zscore');
distances4 = distances4.';

distances5 = distances5.';
distances5 = normalize(distances5,'zscore');
distances5 = distances5.';

distances6 = distances6.';
distances6 = normalize(distances6,'zscore');
distances6 = distances6.';

distances7 = distances7.';
distances7 = normalize(distances7,'zscore');
distances7 = distances7.';

distances8 = distances8.';
distances8 = normalize(distances8,'zscore');
distances8 = distances8.';

distances9 = distances9.';
distances9 = normalize(distances9,'zscore');
distances9 = distances9.';

distances10 = distances10.';
distances10 = normalize(distances10,'zscore');
distances10 = distances10.';

distances11 = distances11.';
distances11 = normalize(distances11,'zscore');
distances11 = distances11.';

distances12 = distances12.';
distances12 = normalize(distances12,'zscore');
distances12 = distances12.';

distances13 = distances13.';
distances13 = normalize(distances13,'zscore');
distances13 = distances13.';

distances14 = distances14.';
distances14 = normalize(distances14,'zscore');
distances14 = distances14.';

distances15 = distances15.';
distances15 = normalize(distances15,'zscore');
distances15 = distances15.';

distances16 = distances16.';
distances16 = normalize(distances16,'zscore');
distances16 = distances16.';

distances17 = distances17.';
distances17 = normalize(distances17,'zscore');
distances17 = distances17.';

executionsID = times(:,1);
duration = times(:,2);
duration = normalize(duration,'zscore');


% Cluster analysis

data = [duration,distances1,distances2,distances3,distances4,...
        distances5,distances6,distances7,distances8,distances9,...
        distances10,distances11,distances12,distances13,distances14...
        distances15,distances16,distances17];

    
    
% Cluster parameters

% all variables (std != 0)
epsilon = 3;
minpts = 25;

    
    
[idx] = dbscan(data,epsilon,minpts);



% indexes = idx==-1;
%  
% cluster = executionsID(indexes);
% writematrix(cluster,'ouliers_all_variables.csv');


indexes = idx==1;
 
cluster = executionsID(indexes);
writematrix(cluster,'cluster_all_variables_3.csv');




















% Plot clusters
% gscatter(executionsID,duration,idx,'rb','..',20);



% Tridimensional plot cluster anlysis

clusters = unique(idx);

c = ['r','b','g'];
index = 1;

for i=clusters.'
   
    % 1,5,6
    subplot(1,2,1);
    %subplot('Position',pos1)
    plot3(data(idx==i,5),data(idx==i,6),data(idx==i,1),...
    '.','Color',c(index),'MarkerSize',20,'DisplayName',char("cluster " + string(index)));
    hold on;  
    xlabel('Variable 5');
    ylabel('Variable 6');
    zlabel('Variable 1');
    
    index = index+1;
    
end

legend('data outliers', 'cluster 1')





% plot means

indexes = idx==-1;

cluster = executionsID(indexes);
%writematrix(cluster,'cluster_all_variables.csv');

good_executions = data(cluster,:);
exe = median(good_executions);

subplot(1,2,2);
%subplot('Position',pos2)
plot(1:length(exe),exe,'rx','MarkerSize',10,...
     'LineWidth',1.8,'DisplayName','outliers execution median');
xlim([0 19])

xticks(1:18);

xlabel('variables')
ylabel('dtw value')

hold on

indexes = idx==1;

cluster = executionsID(indexes);

good_executions = data(cluster,:);
exe = median(good_executions);

plot(1:length(exe),exe,'bx','MarkerSize',10,...
     'LineWidth',1.8,'DisplayName','good execution median');
xlim([0 19])

legend('data outliers', 'cluster 1')


% Comparation variables

% figure

% ind = 1;


% for i=cluster.'
%    
%     exe = data(i,:);
%     plot(1:length(exe),exe,'.','MarkerSize',20,...
%          'DisplayName',char("execution" + string(cluster(ind))));
%     xlabel('variables')
%     ylabel('dtw value')
%     hold on; 
%     
%     ind = ind + 1;
%         
% end


 
 