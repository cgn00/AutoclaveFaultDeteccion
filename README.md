GPT3 Chat Bot, [5/2/2023 10:28 PM]
# Project Title: Executions Analyzer

This project provides data exploration, manipulation, and clustering capabilities for a multi-variable execution dataset. It contains functionalities to explore the executions, group executions by their phase, and discover insights that might help improve the processes.

## Libraries 
- Pandas
- Numpy
- Logging
- Os
- Json
- Datetime
- Sklearn
- Collections
- Dtaidistance
- Plotly
- Matplotlib
- Seaborn

Use `pip install <library_name>` to install these libraries.

## Class

This project has a single class called `executions_analyzer`. The class has various methods for calling different functionalities of the project.

### Functions

- `init()`: Initializes the class by setting the directory paths to the data and other required configurations.
- `load_phase_conf_json()`: This function loads the phase configuration saved as JSON, and initialize the construct of each sequence config and adds it to a list of sequences config.
- `logger_config()`: Configures the logger.
- `visualize_single_phase()`: Plots the data of a single phase in plotly and matplotlib visualizations.
- `build_samples_for_cluster()`: Builds samples for executions clusters. It groups the executions and removes the executions with duration time=0.
- `apply_cluster_algorithm()`: Applies a clustering algorithm to the provided sequence's sample data (variables).
- `apply_classification()`: It takes two samples (list of points in the same sequence) and returns the DTW distance.

GPT3 Chat Bot, [5/2/2023 10:28 PM]
- `calculate_phase_durations()`: It calculates the duration of each phase in each sequence.
- `phase_summary()`: Produces a dataframe from the phase durations with summaries on means, standard deviations, etc.
- `group_two_phases(name_a, name_b, criterion)`: This function groups two phases into a single phase, based on a criterion. The function can either exclude partial executions, the state of the variables, or both from the groupings.
- `visualize_states_of_two_columns(column_a, column_b, point_id, name_phase)`: Produces a scatter plot that visualizes the data of the two variables(columns) by point_id and displays the results by each phase(column name). It also adds information about the average of each phase (column) in a vertical line across the chart.
- `visualize_states_of_two_columns_all_points(column_a, column_b)`: Produces a scatter plot that shows the data of the two variables(columns) by point_id for all phases.
- `visualize_variables_states(name_phase)`: Produces a scatter plot that shows the data of all variables in a phase. The plot has a time on the horizontal axis and the variable values on the vertical axis.

## Conclusion
The `executions_analyzer` class is a valuable tool for exploring the executions, aggregating all the executions by a given phase. You can also build a clustering model, apply classification, produce visualizations, and calculate phase durations. All of these functionalities are useful for analyzing complex datasets. By following the instr

GPT3 Chat Bot, [5/2/2023 10:28 PM]
uctions outlined in this README, you should be able to use the project with ease.
