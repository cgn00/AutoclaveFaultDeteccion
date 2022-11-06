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

            //RUL_Prediction_NN.analysis.Run("Presurización");
            
            RUL_Prediction_NN.analysis.CleanExecutionCSV();

            RUL_Prediction_NN.analysis.SplitSequences();

            Console.WriteLine("Finish");
            Console.ReadLine();


        }

    }
}