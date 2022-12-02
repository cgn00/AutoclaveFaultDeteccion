// See https://aka.ms/new-console-template for more information
using System;
using System.Collections.Generic;
using System.Globalization;
using Tensorflow;
using Tensorflow.NumPy;
using System.Linq;
using Accord.Math;
using Accord.Statistics;
using Accord.Collections;
using Accord.Math.Differentiation;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Accord.Statistics.Kernels;
using Tensorflow.Keras.Engine;
using System.IO;
using RUL_Prediction_NN.Transformation;
using MathNet.Numerics.Differentiation;
using MathNet.Numerics.Statistics;
using RUL_Prediction_NN.Model;
using RUL_Prediction_NN.Train;
using RUL_Prediction_NN.CMAPSS;
using RUL_Prediction_NN.Produccion_principal;
using RUL_Prediction_NN.Recirculacion;

namespace AutoclaveFailDetection
{
    public class Program
    {
        static void Main(string[] args)
        {
<<<<<<< HEAD
            RUL_Prediction_NN.Analysis.LoadBaseDirectory();
            
            RUL_Prediction_NN.Analysis.CleanExecutionCSV();
=======
            RUL_Prediction_NN.analysis.CleanExecutionCSV();
>>>>>>> origin/preprocessing_without_mlnet

            // RUL_Prediction_NN.Analysis.SplitSequences();

            foreach (var sequence in RUL_Prediction_NN.analysis.sequencesName)
            {
                RUL_Prediction_NN.analysis.sequence_directory = RUL_Prediction_NN.analysis.base_directory + sequence + @"\";
                RUL_Prediction_NN.analysis.phases_by_sequence_directory = sequence + @"_phases.csv";
                Console.WriteLine(RUL_Prediction_NN.analysis.sequence_directory);
                Console.WriteLine(RUL_Prediction_NN.analysis.phases_by_sequence_directory);

                //RUL_Prediction_NN.analysis.PreprocessingForClusteringByTimeAndSamplesInPython(); //Run this function before obtain the executions IDs by time and #s of samples
                RUL_Prediction_NN.analysis.RunAllPhases(); //Run this function after obtain the executions IDs by time and #s of samples
            }


            Console.WriteLine("Finish");
            Console.ReadLine();

        }

    }
}