using Accord.Math;
using Microsoft.Data.Analysis;
using Microsoft.ML;
using RUL_Prediction_NN.Data;
using RUL_Prediction_NN.data_model;
using System.Data;
using System.Reflection;
using System.Linq;
//using Accord.Statistics.Kernels;
using Tensorflow;
using Tensorflow.Contexts;
using Tensorflow.NumPy;
using Microsoft.Extensions.Logging;
using log4net;
using log4net.Config;

namespace RUL_Prediction_NN
{

    public static class Analysis
    {
        static List<int> n_variables = new List<int>() {7, 8, 9, 10, 11, 12, 13 }; // 

        static int index = 0; //variable created to record de index of the serch in the function: GetExecutionByID

        private static readonly ILog _log = LogManager.GetLogger(typeof(Analysis));

        // Base directory for save and read results of analysis
        static string base_directory = @"D:\CGN\projects\AutoclaveFailDeteccion\data";
        static readonly string conductivity_directory = @"\Conductivities\";
        static readonly string duration_directory = @"\Durations\";
        static readonly string variables_directory = @"\Variables\";
        static readonly string data_directory = @"\Datas\";
        static readonly string samples_directory = @"\Samples For Executions Cluster\";
        static readonly string phases_directory = @"\Datos\phases_to_analysis.csv";
        //static string phases_directory = @"\Datos\phases.csv";
        static readonly string executions_directory = @"Datos\clean_executions.csv";
        static readonly string executions_to_filter = @"Datos\executionsNaN.csv";
        static string? sequence_directory;
        static string? phases_by_sequence_directory;


        /*
              Main functions
        */
        public static void LoadBaseDirectory()
        {
            ///This functions is to load the Base Directory string to read and save the results of the Execution of the pogram
            ///it's necesary to have a file: baseDirectory.txt with the base directory, located at:..\source\repos\AutoclaveFailDetection
            
            try
            {
                var reader = new StreamReader("baseDirectory.txt");        //Configuracion del objeto de lectura
                base_directory = reader.ReadToEnd();
                reader.Close();
            }
            catch(Exception e)
            {
                //_log.Error(e.Message);
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }
            Console.WriteLine(base_directory);
        }

        public static void CleanExecutionCSV()
        {
            ///this function is to delete executions with incorrect Start or End Date
            
            string path_to_save = base_directory + executions_directory;

            if(File.Exists(path_to_save))
            {
                return;
            }

            var executionsList = new List<Execution>();
            var clean_executions = new List<Execution>();

            List<string> headers = new List<string> { "SequenceId","ExecutionId","StartTime","EndTime","StartOpId","EndOperatorId","SequenceName" };

            try
            {
                var mlContext = new MLContext();
                IDataView data = mlContext.Data.LoadFromTextFile<Execution>(base_directory + executions_to_filter, separatorChar: ',', hasHeader: true);
                var dataList = mlContext.Data.CreateEnumerable<Execution>(data, false);

                var df = data.ToDataFrame(dataList.Count());
                

                executionsList = DataFrameToList<Execution>(df);
                //var dtCheck = df.Columns["StartDate"].ElementwiseLessThanOrEqual<DataFrameColumn>(df.Columns["EndDate"]);
                
                //var df = DataFrame.LoadCsv(base_directory + executions_to_filter);
                //df.FillNulls(0, inPlace: true);

                //executions = pd.read_executions(base_directory + executions_to_filter, headers: true, sep: ',');
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }
            
            if(executionsList[0].StartDate < executionsList[0].EndDate)
            {
                clean_executions.Add(executionsList[0]);
            }

            for(int i = 1; i<executionsList.Count; i++)
            {
                if(executionsList[i].StartDate >= clean_executions[clean_executions.Count - 1].EndDate && executionsList[i].StartDate < executionsList[i].EndDate)
                {
                    //this conditions is to delete executions with incorrect Date
                    clean_executions.Add(executionsList[i]);
                }
            }

            try
            {
                pd.executions_to_csv(path_to_save, sep: ",", headers:headers, columns:clean_executions, append: false);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }

            Console.WriteLine("Executions.Count = {0}", executionsList.Count);
            Console.WriteLine("Clean_Executions.Count = {0}", clean_executions.Count);
        }

        public static void SplitSequences()
        {
            var executions = new List<Execution>();
            var phases = new List<Phase>();
            try
            {
                executions = pd.read_executions(base_directory + executions_directory, headers: true, sep: ',');
                phases = pd.read_phases(base_directory + phases_directory, headers: true, sep: ',');
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }

            var sequencesName = new List<string>();
            var executions_id_by_sequences = new List<List<int>>();

            foreach(var execution in executions)
            {
                if (!sequencesName.Contains(execution.Name))
                {
                    sequencesName.Add(execution.Name);
                }
            }

            foreach(var name in sequencesName)
            {
                var temp_executions_id = new List<int>();
                //Console.WriteLine("temp_executions_id count = {0}", temp_executions_id.Count);

                foreach(var execution in executions)
                {
                    if(execution.Name == name)
                    {
                        temp_executions_id.Add(execution.ExecutionId);
                    }
                }
                executions_id_by_sequences.Add(temp_executions_id);
                Console.WriteLine("{0}: executions count = {1}", name, temp_executions_id.Count());
                //Console.WriteLine("executions founded = {0}", temp_executions_id.Count);
                //temp_executions_id.Clear();
            }

            int i = 0;
            int exec_not_conteined = 0;

            var phases_not_contained = new List<Phase>();

            foreach(var phase in phases)
            {
                phases_not_contained.add(phase);
            }
            
            foreach(var exec_ids in executions_id_by_sequences)
            {
                var temp_phases = new List<Phase>();

                foreach(var phase in phases)
                {
                    if(exec_ids.Contains(phase.ExecutionId))
                    {
                        temp_phases.Add(phase);
                        phases_not_contained.Remove(phase);
                        //Console.WriteLine("{0}", phases_not_contained.Remove(phase));
                    }
                }
                var path = base_directory + @"\" + sequencesName[i] + @"\";
                
                if(!Directory.Exists(path))
                {
                    Directory.CreateDirectory(path);
                }

                try
                {
                    pd.phases_to_csv(path + sequencesName[i] + "_phases.csv", sep: ",", columns: temp_phases, append:false);
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    Console.WriteLine("Press to exit...");
                    Console.ReadKey();
                    return;
                }
                i += 1;
            }

            Console.WriteLine("phases not contained = {0}", phases_not_contained.Count);
            if (!Directory.Exists(base_directory))
            {
                Directory.CreateDirectory(base_directory);
            }
            try
            {
                pd.phases_to_csv(base_directory + "phases_not_contained.csv", sep: ",", columns: phases_not_contained, append:false);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }

            var phases_by_executions = new List<Phase>();

            Console.WriteLine("phases count = {0}", phases.Count);
            i = 0;

            var test_ids = new List<int>();
            
            //printing the executions id not containned in executions.csv but containned in phases_to_analisys.csv
            //foreach(var phase in phases)
            //{
            //    if(!test_ids.Contains(phase.ExecutionId))
            //    {
            //        test_ids.Add(phase.ExecutionId);
            //        //Console.WriteLine(phase.ExecutionId);

            //    }

            //    if (!executions_id_by_sequences[0].Contains(phase.ExecutionId) && !executions_id_by_sequences[1].Contains(phase.ExecutionId))
            //    {
            //        Console.WriteLine(phase.ExecutionId);
                    
            //    }
            //}

            //Console.WriteLine("test_ids count = {0}", test_ids.Count);

            Console.WriteLine("Executions count = {0}", executions.Count);

            foreach(var sequence in sequencesName)
            {
                sequence_directory = base_directory + sequence + @"\";
                phases_by_sequence_directory = sequence + @"_phases.csv";
                Console.WriteLine(sequence_directory);
                Console.WriteLine(phases_by_sequence_directory);
                
                RunAllPhases();
            }
        }

        public static void RunAllPhases()
        {
            //filterPhaseName(base_directory + @"Datos\phases.csv");
            
            var phases = new List<Phase>();
            try
            {
                phases = pd.read_phases(sequence_directory + phases_by_sequence_directory);
                
            }
            catch(Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }

            var phase_name = new List<string>();

            for(int i = 0; i< phases.Count; i++)
            {
                var phase = phases[i].Text;
                bool pass = phase_name.Contains(phase);
               
                if (!pass)
                {
                    phase_name.Add(phases[i].Text);
                }
            }

            foreach(string p_name in phase_name)
            {
                Console.WriteLine(p_name);
               
                Run(p_name);
            }
            Console.WriteLine("--------------------");
            
        }

        public static void Run(string phase_name)
        {

            /*
             *  Main function
             */

            /*

            // Save all values of conductivities
            SaveConductivitiesSpecificPhase(phase_name);

            // Save indexes of correct executions according the process
            DetectCorrectExecutions(phase_name);
            DetectCorrectExecutions(phase_name);

            // Save values of correct conductivities
            SaveConductiviesForQuality(phase_name, quality: "good_executions");

            // Save durations of correct conductivities
            SaveDurationOfSpecificPhase(phase_name);
            
            */

            var executions_count = FilterSamplesForPhases(phase_name);

            SavePhaseDuration(phase_name);

            
            OrganizedSamplesForVariables(phase_name, total_executions: executions_count);
            
            GetDatasForTraining(phase_name, variables: n_variables, n_executions: executions_count);
            GetLabelsForTraining(phase_name, n_executions: executions_count);
            
            // Save time series of executions for each variable
            for(int i = 1; i <= n_variables.Count; i++)
            {
                Console.WriteLine("Saving Variable {0}" , n_variables[i - 1]);
                SaveVariable(phase_name, i);
            }
            
        }

        //private static void filterPhaseName(string dir)
        //{
        //    var phases = new List<Phase>();

        //    try
        //    {
        //        phases = pd.read_phases(base_directory + @"Datos\phases.csv");
        //    }

        //    catch (Exception e)
        //    {
        //        Console.WriteLine(e.Message);
        //        Console.WriteLine("Press to exit...");
        //        Console.ReadKey();
        //        return;
        //    }
            
        //    for(int i = 0; i < phases.Count; i++)
        //    {
        //        if(phases[i].Text.Equals("\"Esterilización \""))
        //        {
        //            phases[i].Text = "Esterilización";
        //        }
        //        else if(phases[i].Text.Equals("Llenado"))
        //        {
        //            phases[i].Text = "LLenado";
        //        }
        //    }
            
        //    try
        //    {
        //        string direc = base_directory + @"Datos\phases_to_analysis.csv";
        //        List<string> head = new List<string>() { "EntityId", "ExecutionId", "Time", "Text" };
        //        var entity = new List<string>();
        //        var execution = new List<string>();
        //        var time = new List<string>();
        //        var text = new List<string>();
                
        //        for (int i = 0; i < phases.Count; i++)
        //        {
        //            entity.Add(phases[i].EntityId.ToString());
        //            execution.Add(phases[i].ExecutionId.ToString());
        //            time.Add(phases[i].Time.ToString());
        //            text.Add(phases[i].Text.ToString());
        //        }
        //        /*
        //        for (int i = 0; i < phases.Count; i++)
        //        {
        //            arrays[i] = new string[4];
        //            //arrays[i][0] = "hello";
        //            arrays[i][0] = phases[i].EntityId.ToString();
        //            arrays[i][1] = phases[i].ExecutionId.ToString();
        //            arrays[i][2] = phases[i].Time.ToString();
        //            arrays[i][3] = phases[i].Text;
                    
        //        }*/

        //        string[][] arrays = new string[][] { entity.ToArray(), execution.ToArray(), time.ToArray(), text.ToArray() };
                
        //        var transp = arrays.Transpose();
        //        var phases_to_save = np.array(transp.ToMatrix(transpose:false));

        //        pd.to_csv(direc, headers:head, columns:phases_to_save, type:TypeCode.Double);
        //        Console.WriteLine("phases_to_analysis.csv is saved");
        //    }

        //    catch (Exception e)
        //    {
        //        Console.WriteLine(e.Message);
        //        Console.WriteLine("Press to exit...");
        //        Console.ReadKey();
        //        return;
        //    }
            
        //}




        static void SaveConductivitiesSpecificPhase(string phase_name)
        {
            /*
             * Save samples of variable 'conductivity' in csv file
             */


            if (File.Exists(base_directory + phase_name + conductivity_directory + "indexes.csv") && File.Exists(base_directory + phase_name + conductivity_directory + "conductivities.csv"))
            {
                return;
            }

            var phases_times = GetLimitsTimesOfPhase(phase_name);

            var s1 = new List<Sample>();
            var s2 = new List<Sample>();
            var s3 = new List<Sample>();

            try
            {
                s1 = pd.read_samples(@".\JTData\Samples_4-6.csv");
                s2 = pd.read_samples(@".\JTData\Samples_7-8.csv");
                s3 = pd.read_samples(@".\JTData\Samples_9-10.csv");
            }

            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }

            var samples = new List<Sample>();
            samples.AddRange(s1);
            samples.AddRange(s2);
            samples.AddRange(s3);

            // Divide samples for phases
            var sam = GetSamplesDividedByPhasesAndVariable(samples, phase_name, variable_id: 12);

            var conductivities = new List<List<double>>();

            // Write conductivities in file           
            foreach (var s_1 in sam)
            {
                var temp = new List<double>();

                foreach (var s_2 in s_1)
                {
                    temp.Add(s_2.Value);
                }

                conductivities.Add(temp);
                
            }

            if (!Directory.Exists(base_directory + phase_name + conductivity_directory))
            {
                Directory.CreateDirectory(base_directory + phase_name + conductivity_directory);
            }

            foreach (var c in conductivities)
            {
                pd.to_csv(base_directory + phase_name + conductivity_directory + "conductivities.csv", columns: c.ToArray(), type: TypeCode.Double);
            }

            var indexes = new List<int>();

            // Save indexes of samples

            for (int i = 0; i < phases_times.Count; i++)
            {
                indexes.Add(i);
            }

            pd.to_csv(base_directory + phase_name + conductivity_directory + "indexes.csv", columns: np.array(indexes.ToArray()).reshape((-1, 1)), type: TypeCode.Int32, append: false);

            Console.WriteLine("Conductivities values are saved!");

        }

        public static void DetectCorrectExecutions(string phase_name)
        {

            if (!Directory.Exists(base_directory + phase_name + conductivity_directory))
            {
                Console.WriteLine("Directory invalid");
                return;
            }

            if (File.Exists(base_directory + phase_name + conductivity_directory + "conductivities_good_executions.csv"))
            {
                Console.WriteLine("The file exist!");
                return;
            }

            var conductivity = new List<List<double>>();
            var indexes = new List<float>();

            try
            {
                conductivity = pd.read_samples_of_executions(base_directory + phase_name + conductivity_directory + "conductivities.csv");
                indexes = pd.read_csv(base_directory + phase_name + conductivity_directory + "indexes.csv").Reshape().ToList();
            }

            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }

            var good_executions = new List<int>();

            // Select executions
            for (int i = 0; i < conductivity.Count; i++)
            {
                // (Initial conductivity > 1.4 and final conductivity < 1.4)
                if (phase_name == "Recirculación")
                {
                    // For executions without samples (low duration)
                    if (conductivity[i].Count != 0)
                    {
                        // Analyze if conductivty increase the thereshold in at moment (1.4)
                        if (conductivity[i][0] <= 1.4)
                        {

                            for (int j = 0; j < conductivity[i].Count; j++)
                            {
                                // Analyze if conductivity finalize under thereshold
                                if (conductivity[i][j] > 1.4)
                                {
                                    if (conductivity[i][conductivity[i].Count - 1] <= 1.4)
                                    {
                                        good_executions.Add((int)indexes[i]);
                                        break;
                                    }

                                    else
                                    {
                                        break;
                                    }
                                }


                            }

                        }

                        // Analize if executions start with high conductivity and finish with low conductivity
                        else if (conductivity[i][0] > 1.4 && conductivity[i][conductivity[i].Count - 1] <= 1.4)
                        {
                            good_executions.Add((int)indexes[i]);
                        }

                    }


                }

                // (Final conductivity > 1.3) 
                else if (phase_name == "Producción principal")
                {
                    // For executions without enough samples (low duration)
                    if (conductivity[i].Count >= 10)
                    {
                        // Analize if start with low conductivity and finish with high conductivity
                        if (conductivity[i][conductivity[i].Count - 1] >= 1.29 && conductivity[i][0] < 1.4 && conductivity[i][conductivity[i].Count - 1] <= 1.35)
                        {
                            good_executions.Add((int)indexes[i]);
                        }

                    }
                }

                else
                {
                    Console.WriteLine("Invalid phase name!");
                    return;
                }

            }


            try
            {
                pd.to_csv(base_directory + phase_name + conductivity_directory + "conductivities_good_executions.csv", columns: np.array(good_executions.ToArray()).reshape((-1, 1)), append: false, type: TypeCode.Int32);
                Console.WriteLine("Detected correct executions!");
            }

            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }

        }

        public static void SaveConductiviesForQuality(string phase_name, string quality)
        {
            /*
             *  Save conductivities for specific quality
             *  quality = {initial_low, initial_high, final_low, final_high}    
             */

            if (!Directory.Exists(base_directory + phase_name + conductivity_directory))
            {
                Console.WriteLine("Directory invalid");
                return;
            }

            if (File.Exists(base_directory + phase_name + conductivity_directory + "values_conductivities_" + quality + ".csv"))
            {
                return;
            }

            var conductivity = new List<List<double>>();
            var indexes = new List<float>();

            try
            {
                conductivity = pd.read_samples_of_executions(base_directory + phase_name + conductivity_directory + "conductivities.csv");
                indexes = pd.read_csv(base_directory + phase_name + conductivity_directory + "conductivities_" + quality + ".csv").Reshape().ToList();
            }

            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }

            var good_executions = new List<List<double>>();

            foreach (var i in indexes)
            {
                var temp = conductivity[(int)i];

                good_executions.Add(temp);
            }

            // Save conductivities values

            try
            {
                foreach (var c in good_executions)
                {
                    pd.to_csv(base_directory + phase_name + conductivity_directory + "values_conductivities_" + quality + ".csv", columns: c.ToArray(), type: TypeCode.Double);
                }

                Console.WriteLine("Conductivities values of correct executions are saved!");
            }

            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }


        }

        
        static void SavePhaseDuration(string phase_name)
        {
            var directory = sequence_directory + @"Samples Sorts by Phases\" + phase_name ;

            if (File.Exists(directory + "durations.csv"))
            {
                return;
            }

            var indexes = new List<float>();

            try
            {
                indexes = pd.read_csv(directory + @"\executions_with_samples.csv").Reshape().ToList();
            }
            catch(Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }

            var times = GetLimitsTimesOfPhase(phase_name);
            Console.WriteLine("times.count = {0}", times.Count);
            var durations_times = new List<double>();


            for (int i = 0; i < indexes.Count; i++)
            {
                var sample = new List<Sample>();
                try
                {
                    //Console.WriteLine(directory + @"\Samples For Executions Cluster\" + "executions_" + (i + 1));
                    sample = pd.read_samples(directory + @"\Samples For Executions Cluster\" + "executions_" + indexes[i] + ".csv", partial:true);
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    Console.WriteLine("Press to exit...");
                    Console.ReadKey();
                    return;
                }
                //durations_times.Add((sample[sample.Count - 1].Time - sample[0].Time).TotalMinutes);
                durations_times.Add((times[(int)indexes[i] - 1].Item2 - times[(int)indexes[i] - 1].Item1).TotalMinutes);   
            }

            var to_save = durations_times.ToArray().Transpose();
          

            if (!Directory.Exists(directory + duration_directory))
            {
                Directory.CreateDirectory(directory + duration_directory);
            }

            try
            {
                pd.to_csv(directory + duration_directory + "durations.csv", columns: to_save, type: TypeCode.Double, append: false);
                Console.WriteLine("Durations of executions are saved!");
            }

            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }

            
        }

        static void SaveDurationOfSpecificPhase(string phase_name)
        {

            /*
             *  Save executions duration (in minutes) for specific phase
             *  Divided in conductivity cuality (initial low/hight or final low/high or good_executions criteria)
             */


            if (File.Exists(base_directory + phase_name + duration_directory + "durations_good_executions.csv"))
            {
                return;
            }

            var conductivities = new List<List<double>>();
            var indexes = new List<float>();

            var times = GetLimitsTimesOfPhase(phase_name);

            try
            {
                conductivities = pd.read_samples_of_executions(base_directory + phase_name + conductivity_directory + "values_conductivities_good_executions.csv");
                indexes = pd.read_csv(base_directory + phase_name + conductivity_directory + "conductivities_good_executions.csv").Reshape().ToList();
            }

            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }

            var conductivity = new List<double>();
            var durations_times = new List<double>();
            var indexes_phases = new List<double>();

            if (phase_name == "Recirculación")
            {

                for (int i = 0; i < conductivities.Count; i++)
                {
                    conductivity.Add(conductivities[i][0]);
                    durations_times.Add((times[(int)indexes[i]].Item2 - times[(int)indexes[i]].Item1).TotalMinutes);
                    indexes_phases.Add(indexes[i]);
                }

            }

            else if (phase_name == "Producción principal")
            {
                for (int i = 0; i < conductivities.Count; i++)
                {
                    conductivity.Add(conductivities[i][conductivities[i].Count - 1]);
                    durations_times.Add((times[(int)indexes[i]].Item2 - times[(int)indexes[i]].Item1).TotalMinutes);
                    indexes_phases.Add(indexes[i]);
                }
            }

            else
            {
                Console.WriteLine("Invalid phase name!");
                return;
            }

            var arrays = new double[][] { indexes_phases.ToArray(), conductivity.ToArray(), durations_times.ToArray() };
            var to_save = np.array(arrays.ToMatrix(transpose: true));


            if (!Directory.Exists(base_directory + phase_name + duration_directory))
            {
                Directory.CreateDirectory(base_directory + phase_name + duration_directory);
            }

            try
            {
                pd.to_csv(base_directory + phase_name + duration_directory + "durations_good_executions.csv", columns: to_save, type: TypeCode.Double, append: false);
                Console.WriteLine("Durations of executions are saved!");
            }

            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }


        }

        static int FilterSamplesForPhases(string phase_name, string discriminate = null, bool maxim = false)
        {
            /*
             *  Filter samples for executions of specific phase
             *  Saving on independent csv files
             *  
             *  discriminate = dtw indicate that analysis of distances between signals criteria is used for descriminate good executions
             *  discriminate = clusters indicate that analysis of cluster criteria is used for descriminate good executions
             *  maxim indicate that initiation of execution is maxim value of conductivity (Recirculación)
             */
            var directory = sequence_directory + @"Samples Sorts by Phases\" + phase_name + samples_directory;
            
            if (Directory.Exists(directory))
            {
                //string[] files = Directory.GetFiles(directory,"*.csv");

                //return files.Length;

                return 0;
            }
            


            // Read all samples
            var samples = new List<Sample>();
          
            try
            {
                samples = pd.read_samples(base_directory + @"Datos\samples.csv");
            }

            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return 0;
            }

            // aparecen id: 17...23 q corresponden del 7..13
            for(int i = 0; i < samples.Count; i++)
            {
                if(!n_variables.Contains(samples[i].VariableId))
                {
                    samples[i].VariableId -= 10;
                }
            }

            // Get limits times of executions of specific phase
            var phases_times = GetLimitsTimesOfPhase(phase_name);

            var new_phases_times = new List<(DateTime, DateTime)>();

            /*
            // Redefine limits times

            if (discriminate == "dtw")
            {

                var index_dtw = new List<float>();

                try
                {
                    index_dtw = pd.read_csv(base_directory + phase_name + conductivity_directory + "conductivities_dtw.csv").Reshape().ToList();
                }

                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    Console.WriteLine("Press to exit...");
                    Console.ReadKey();
                    return 0;
                }


                foreach (var i in index_dtw)
                {
                    new_phases_times.Add((phases_times[(int)i].Item1, phases_times[(int)i].Item2));
                }

            }

            else if (discriminate == "clusters")
            {

                var index_cluster = new List<float>();

                try
                {
                    index_cluster = pd.read_csv(base_directory + phase_name + duration_directory + "conductivities_cluster.csv").Reshape().ToList();
                }

                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    Console.WriteLine("Press to exit...");
                    Console.ReadKey();
                    return 0;
                }

                foreach (var i in index_cluster)
                {
                    new_phases_times.Add((phases_times[(int)i].Item1, phases_times[(int)i].Item2));
                }

            }

            else
            {

                var indexes = new List<float>();

                try
                {
                    indexes = pd.read_csv(base_directory + phase_name + conductivity_directory + "conductivities_good_executions.csv").Reshape().ToList();
                }

                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    Console.WriteLine("Press to exit...");
                    Console.ReadKey();
                    return 0;
                }

                foreach (var i in indexes)
                {
                    new_phases_times.Add((phases_times[(int)i].Item1, phases_times[(int)i].Item2));
                }


            }
            */

            // Get samples of phases for determinates intervals of times
            var sam = GetSamplesDividedByPhasesAndVariable(samples, phase_name);


            // Save samples
            //float[,] indexes;
            var indexes = new List<float>();
            try
            {
                string path = sequence_directory + @"Samples Sorts by Phases\" + phase_name + @"\executions_with_samples.csv";
                indexes = pd.read_csv(path).Reshape().ToList();
            }
            catch(Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
            }

            Directory.CreateDirectory(directory);

            for (int i = 0; i < sam.Count; i++)
            {
                var sorted_samples = new List<Sample>();
                sorted_samples = SortAscendingTimeSeries(sam[i]);
                try
                {
                    pd.samples_to_csv(directory + @"executions_" + indexes[i] + ".csv", columns: sorted_samples);
                }

                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    Console.WriteLine("Press to exit...");
                    Console.ReadKey();
                    return 0;
                }

            }

            Console.WriteLine("Correct executions: {0}", sam.Count);


            // Count of executions
            return sam.Count;

        }

        static void OrganizedSamplesForVariables(string phase_name, int total_executions)
        {

            /*
             *  Divide samples of executions per variables
             *  Saving on independent csv files
             *  
             *  total_executions indicate the number of good executions
             */



            if (total_executions == 0)
            {
                Console.WriteLine("There isn't any executions");
                return;
            }

            var index = total_executions;
            var directory = sequence_directory + @"Samples Sorts by Phases\" + phase_name + samples_directory;

            var indexes = new List<float>();

            try
            {
                string path = sequence_directory + @"Samples Sorts by Phases\" + phase_name + @"\executions_with_samples.csv";
                indexes = pd.read_csv(path).Reshape().ToList();
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }

            for (int i = 1; i < index + 1; i++)
            {
                //Console.WriteLine(i);
                var sample = new List<Sample>();


                try
                {
                    sample = pd.read_samples(directory + "executions_" + indexes[i-1] + ".csv", headers: false, partial: true);
                }

                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    Console.WriteLine("Press to exit...");
                    Console.ReadKey();
                    return;
                }


                var dir = Directory.CreateDirectory(directory + "execution_" + i + @"\");
                OrderSamplesByVariables(sample, dir.FullName);

            }

            Console.WriteLine("Samples of executions are saved!");

        }

        static void SaveVariable(string phase, int var)
        {
            /*
             * Save time series of variables in executions
             */

            if(Directory.Exists(sequence_directory + @"Samples Sorts by Phases\" + phase + variables_directory + "variable" + (n_variables[var-1]) + ".csv"))
            {
                Console.WriteLine("The variables are already saved");
                return;
            }

            float[,] datas;

            try
            {
                datas = pd.read_csv(sequence_directory + @"Samples Sorts by Phases\" + phase + data_directory + @"data.csv");
            }

            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }


            var index = datas.ToJagged().Distinct()[0];

            var variable = new List<List<float>>();
            //var indexes = new List<double>();

            // variable 1 -> duration

            foreach (var i in index)
            {
                var seq = from x in datas.ToJagged()
                          where x[0] == i
                          select (x);

                variable.Add(seq.ToArray().GetColumn(var + 1).ToList());
                //indexes.Add(i);

            }


            if (!Directory.Exists(sequence_directory + @"Samples Sorts by Phases\" + phase + variables_directory))
            {
                Directory.CreateDirectory(sequence_directory + @"Samples Sorts by Phases\" + phase + variables_directory);
            }

            foreach (var c in variable)
            {
                pd.to_csv(sequence_directory + @"Samples Sorts by Phases\" + phase + variables_directory + "variable" + (n_variables[var-1]) + ".csv", columns: np.array(c.ToArray()), type: TypeCode.Empty);
            }

            //pd.to_csv(base_directory + phase + variables_directory + "indexes.csv", columns: np.array(indexes.ToArray()).reshape((-1, 1)), type: TypeCode.Double, append: false);

            Console.WriteLine("values of variable {0} are saved!", n_variables[var-1]);

        }







        /*
         *  Datasource
         */


        static void GetDatasForTraining(string phase_name, List<int> variables, int n_executions)
        {

            /*
             *  Build datasource for selected variables
             */

            if (File.Exists(sequence_directory + @"Samples Sorts by Phases\" + phase_name + data_directory + "data.csv"))
            {
                return;
            }

            var directory = sequence_directory + @"Samples Sorts by Phases\" + phase_name + samples_directory;

            var datas = new List<NDArray>();

            for (int i = 1; i <= n_executions; i++)
            {

                var data_temp = new List<double[]>();

                foreach (var v in variables)
                {

                    var temp_samples = new List<Sample>();

                    try
                    {
                        temp_samples = pd.read_samples(directory + "execution_" + (i) + @"\" + v + ".csv", partial: true, headers: false);
                        var t = temp_samples.Count;
                    }

                    catch (Exception e)
                    {
                        Console.WriteLine(e.Message);
                        Console.WriteLine("Press to exit...");
                        Console.ReadKey();
                        return;
                    }

                    // Columns of sequence ID and Time ID before the variables samples columns
                    if (v == variables[0])
                    {
                        data_temp.Add(Enumerable.Repeat(1.0 * i, temp_samples.Count).ToArray());
                        data_temp.Add(DoubleRange(1.0, (double)temp_samples.Count).ToArray());
                    }


                    if (v != 21 && v != 30 && v != 31)
                    {

                        var temp = new List<double>();

                        foreach (var t in temp_samples)
                        {
                            temp.Add(t.Value);
                        }

                        data_temp.Add(temp.ToArray());
                    }

                    // Correct number of samples by frecuency sample
                    else
                    {

                        var temp = new List<double>();

                        for (int j = 0; j < temp_samples.Count; j++)
                        {
                            if (j % 5 == 0)
                            {
                                temp.Add(temp_samples[j].Value);
                            }
                        }


                        // Correct data add or substract sample
                        if (temp.Count < data_temp[0].Length)
                        {
                            temp.Add(temp[temp.Count - 1]);
                        }

                        else if (temp.Count > data_temp[0].Length)
                        {
                            temp.RemoveAt(temp.Count - 1);
                        }

                        data_temp.Add(temp.ToArray());


                    }


                }

                datas.add(np.array(data_temp.ToArray().ToMatrix(true)));
                
            }

            var data_set = np.concatenate(datas.ToArray());

            var data_direc = sequence_directory + @"Samples Sorts by Phases\" + phase_name + data_directory;

            if (!Directory.Exists(data_direc))
            {
                Directory.CreateDirectory(data_direc);
            }

            try
            {
                pd.to_csv(data_direc + "data.csv", columns: data_set, append: false, type: TypeCode.Double);
                Console.WriteLine("Datas for training are saved!");
            }

            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return;
            }




        }

        static void GetLabelsForTraining(string phase_name, int n_executions)
        {

            /*
             *  Labeled data for training neural network
             *  Linear function with remaining time for execution finish
             */

            if (File.Exists(sequence_directory + @"Samples Sorts by Phases\" + phase_name + data_directory + "label.csv"))
            {
                return;
            }

            var directory = sequence_directory + @"Samples Sorts by Phases\" + phase_name + samples_directory;

            var labels = new List<List<double>>();
            var sequences = new List<List<double>>();

            for (int i = 0; i < n_executions; i++)
            {

                var temp = new List<Sample>();

                try
                {
                    temp = pd.read_samples(directory + "execution_" + (i + 1) + @"\" + "12.csv", headers: false, partial: true);
                }

                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    Console.WriteLine("Press to exit...");
                    Console.ReadKey();
                    return;
                }

                var temp_labels = new List<double>();
                var temp_seq = new List<double>();

                for (int j = 0; j < temp.Count; j++)
                {
                    // Total seconds divided by 5 for indicate number of cycles
                    var time = (double)((temp[temp.Count - 1].Time - temp[j].Time).TotalSeconds) / 5;
                    temp_labels.Add(time);

                    temp_seq.Add(i + 1);
                }

                labels.Add(temp_labels);
                sequences.Add(temp_seq);

            }

            if (!Directory.Exists(sequence_directory + @"Samples Sorts by Phases\" + phase_name + data_directory))
            {
                Directory.CreateDirectory(sequence_directory + @"Samples Sorts by Phases\" + phase_name + data_directory);
            }

            // Saving labels
            for (int i = 0; i < sequences.Count; i++)
            {
                var arrays = new double[][] { sequences[i].ToArray(), labels[i].ToArray() };
                var tosave = np.array(arrays.ToMatrix(transpose: true));

                try
                {
                    pd.to_csv(sequence_directory + @"Samples Sorts by Phases\" + phase_name + data_directory + "label.csv", type: TypeCode.Double, columns: tosave);

                }

                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    Console.WriteLine("Press to exit...");
                    Console.ReadKey();
                    return;
                }


            }

            Console.WriteLine("Labels for training are saved!");
            //Console.ReadKey();


        }









        /*
         *  Help functions
         */


        static List<(DateTime, DateTime)> GetLimitsTimesOfPhase(string phase_name)
        {

            /*
             *  Get intervals of times for executions of specific phase
             */

            // Get limits of times executions for specific phase

            var phases = new List<Phase>();
            var phases_by_sequence = new List<Phase>();
            var executions = new List<Execution>();

            try
            {
                //phases = pd.read_phases(base_directory + phases_directory);
                phases_by_sequence = pd.read_phases(sequence_directory + phases_by_sequence_directory, headers:false);
                executions = pd.read_executions(base_directory + executions_directory, headers: true, sep: ',');
            }

            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine("Press to exit...");
                Console.ReadKey();
                return null;
            }
            

            var times = new List<(DateTime, DateTime)>();

            for(int i = 0; i< phases_by_sequence.Count - 1; i++)
            {
                //serach in the list of the phases by sequence the phase that it's being analyzed at this time
                if(phases_by_sequence[i].Text == phase_name)
                {
                    var exec = GetExecutionByID(executions, phases_by_sequence[i].ExecutionId); //obtain the execution where the actual phase belong

                    if (exec != null)
                    {
                        if (phases_by_sequence[i+1].Time > exec.EndDate)//ask if the next phase start after the end Date of the execution that belong the actual phase
                        {
                            //if enter here means that this phase is the last of the execution, then the end of this phase is the end Date of the execution
                            //phase[i] is the last phase of the execution
                            times.Add((phases_by_sequence[i].Time, exec.EndDate));
                        }
                        else
                        {
                            //phase[i] isn't the last phase of the execution, the end of the phase is the start Date of the next phase
                            times.Add((phases_by_sequence[i].Time, phases_by_sequence[i + 1].Time));
                        }
                    }
                    else
                    {
                        Console.WriteLine("Execution ID not found\npress key to exit");
                        Console.ReadKey();
                        return null;
                    }

                }
            }
            /*
             
            var phases_ids_by_sequence = new List<int>();

            foreach (var phase_sequen in phases_by_sequence)
            {
                phases_ids_by_sequence.Add(phase_sequen.ExecutionId);
            }
            
            for ( int i = 0; i < phases.Count-1; i++)
            {
                if (phases[i].Text == phase_name && phases_ids_by_sequence.Contains(phases[i].ExecutionId))
                {
                    times.Add((phases[i].Time, phases[i + 1].Time));
                }
            }
            */
            /*
            for (int i = 0; i < phases.Count; i++)
            {   
                // Analyze if "Recirculación" phase precede to "Producción principal" phase
                if (phase_name == "Recirculación")
                {
                    if (phases[i].Text == phase_name && phases[i + 1].Text == "Producción principal")
                    {
                        times.Add((phases[i].Time, phases[i + 1].Time));
                    }
                }

                // Analyze if "Producción principal" phase precede to "Recirculación" phase
                else if (phase_name == "Producción principal")
                {
                    if (phases[i].Text == phase_name && phases[i + 1].Text == "Recirculación")
                    {
                        times.Add((phases[i].Time, phases[i + 1].Time));
                    }
                }


            }
            */

            return times;

        }

        static Execution GetExecutionByID(List<Execution> executions, int id)
        {
            ///Helper function to obtain the Execution of an especific ID to use to obtain de limits times of a phase
            var exec = new Execution();
            
            for(var i = 0; i < executions.Count; i++)
            {
                if(executions[i].ExecutionId == id)
                {
                    index = i;// save the index, that will be used the next time to start the search at this point 
                    return executions[i];
                }
            }

            return null;
        }

        static List<List<Sample>> GetSamplesDividedByPhasesAndVariable(List<Sample> samples, string phase_name , int variable_id = 0)
        {
            //Console.WriteLine("get");
            List<(DateTime, DateTime)> phases_times = GetLimitsTimesOfPhase(phase_name);

            /*
             *  Get samples divided for phases and variables
             *  
             *  variables_id indicate if divide samples for all or specific variables
             */

            var divided_samples = new List<List<Sample>>();
            var executions_with_samples = new List<int>();

            int phase_time_without_samples = 0;

            for (int i = 0; i < phases_times.Count; i++)
            {
                var temp = new List<Sample>();

                foreach (var s in samples)
                {
                    /*
                    // if the samples are organized by time

                    if(s.Time > phases_times[i].Item1)
                    {
                        if(s.Time <= phases_times[i].Item2)
                        {
                            temp.Add(s);
                            samples.Remove(s); // the next time it's no necesary ask for this sample 
                        }

                        // out of the phase range
                        else
                        {
                            // no more samples for this phase
                            break;
                        }
                    }

                    */

                    
                    // For all variables
                    if (variable_id == 0)
                    {
                        if (s.Time < phases_times[i].Item2 && s.Time >= phases_times[i].Item1)
                        {
                            temp.Add(s);
                        }
                    }

                    // For specific variable
                    else
                    {
                        if (s.Time < phases_times[i].Item2 && s.Time >= phases_times[i].Item1 && s.VariableId == variable_id)
                        {
                            temp.Add(s);
                        }
                    }
                    
                }
                // why not ask if temp isn't null and then add it
                if(temp.Count > 0)
                {
                    divided_samples.Add(temp);
                    executions_with_samples.Add(i+1);
                }
                else
                {
                    phase_time_without_samples++;
                }
                
            }

            var directory = sequence_directory + @"Samples Sorts by Phases\" + phase_name ;
            var to_save = executions_with_samples.ToArray().Transpose();


            if (!File.Exists(directory + @"\executions_with_samples.csv"))
            {
                if(!Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }
                try
                {
                    pd.to_csv(directory + @"\executions_with_samples.csv", columns: to_save, append: false);
                }

                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    Console.WriteLine("Press to exit...");
                    Console.ReadKey();
                    //return 0;
                }
            }
            Console.WriteLine("phase_time_whitout_samples = {0}", phase_time_without_samples);
            //Console.WriteLine(@"DividedSamples.count = {0} \nexecutions = {1}", divided_samples.Count, executions_with_samples.Count);
            return divided_samples;

        }

        static void OrderSamplesByVariables(List<Sample> samples, string directory)
        {

            /*
             *  Order samples by variables and Saving in a csv file
             */
            
            var variables_with_samples = new List<List<Sample>>();

            foreach (var i in n_variables)
            {
                var temp = DivideSamplesForVariables(samples, i);      //return a list of samples of one variable 

                if (temp.Count != 0)
                {
                    //var temp_sorts_samples = SortAscendingTimeSeries(temp);  // sort chronologically the time serie 
                    variables_with_samples.Add(temp);
                }

            }

            foreach (var s in variables_with_samples)
            {
                try
                {
                    pd.samples_to_csv(directory + s[0].VariableId + ".csv", columns: s, append: false);   
                }

                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    Console.WriteLine("Press to exit...");
                    Console.ReadKey();
                    return;
                }

            }

        }

        private static List<Sample> SortAscendingTimeSeries(List<Sample> samples)
        {
            samples.Sort((a, b) => a.Time.CompareTo(b.Time));

            return samples;
        }

        private static List<Sample> DivideSamplesForVariables(List<Sample> samples, int variableId)
        {

            /*
             *  Auxiliar function for divide samples for variables
             */

            var temp = new List<Sample>();

            for (int i = 0; i < samples.Count; i++)
            {
                if (samples[i].VariableId == variableId)
                {
                    temp.Add(samples[i]);
                }

            }

            return temp;
        }

        private static List<T> DataFrameToList<T>(DataFrame df)
        {
            ///Function to convert a Data Frame into a List
            List<T> data = new List<T>(); 
            foreach (var row in df.Rows)
            {
                T item = GetItem<T>(row);
                data.Add(item);
            }
            return data;
        }

        private static T GetItem<T>(DataFrameRow dfr)
        {
            ///Helper function to obtain an Object <T> from a row of a Data Frame
            ///In the Class <T> properties most be declared in the same order that appears in Data Base:
            ///Propertie(i) most be Column(i) 
            Type temp = typeof(T);
            T obj = Activator.CreateInstance<T>();
            
            int i = 0;
            foreach(PropertyInfo pro in temp.GetProperties())
            {
                pro.SetValue(obj, dfr[i], null);
                i++;
            }        
            
            return obj;
        }

        /*
         *  Extensions
         */


        static IEnumerable<double> DoubleRange(double min, double max)
        {

            /*
             *  Range of double values
             */

            for (double value = min; value <= max; value += 1.0)
            {
                yield return value;
            }
        }

        static double[] Resample(double[] source, int n)
        {
            int m = source.Length;
            var destination = new double[n];
            destination[0] = source[0];
            destination[n - 1] = source[m - 1];

            for (int i = 1; i < n - 1; i++)
            {
                var jd = ((double)i * (double)(m - 1) / (double)(n - 1));
                var j = (int)jd;
                destination[i] = (double)Math.Round((source[j] + (source[j + 1] - source[j]) * (jd - (double)j)), 2);
            }

            return destination;

        }
    }
}
