﻿using RUL_Prediction_NN.Data;
using RUL_Prediction_NN.Model;
using RUL_Prediction_NN.Train;
using RUL_Prediction_NN.Transformation;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RUL_Prediction_NN.Produccion_principal
{
    public class mlp_CV_PP
    {


        string directory;
        string model_name;

        public mlp_CV_PP(string model_name = "MLP CV PP")
        {
            this.model_name = model_name;
            directory = @".\results\" + model_name + @"\";
        }

        public void MetricsRun()
        {

            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var RULWarning = 30;

            var (dataframe, labelframe) = DataRead.LoadData();

            var variability = new Variability();

            var wp = 1.0;
            var wm = 1.0;
            var wt = 1.0;
            var theresold = 1.0;
            var window_size = 10;
            int len = 40;

            variability.Fit(directory, wp, wm, wt, theresold, window_size, len);

            dataframe = variability.Transform(dataframe);

            var nomalization = new Zscore();

            nomalization.Fit(dataframe);

            dataframe = nomalization.Transform(dataframe);

            var (data, label, seqs, times) = DataRead.GetSequences(dataframe, labelframe);


            // Configuration model

            var model = new PredictionMLP();

            model.AddInputLayer(((int)data[0].shape[1]));
            model.AddDenseLayer(5, "sigmoid");
            model.AddDenseLayer(1, "linear");



            // Training parameters MSE

            var epochs = 50;
            (string name, int? param) loss_method = ("square_error", null);
            var optimizer = "rmsprop";
            var lr = 0.01f;
            var batch_mode = "continuos";
            var batch_size = 100;
            var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", RULWarning) };
            var data_presentation = "sample";

            var model_name = this.model_name + @"\MSE\";

            // 10 fold Cross-Validation

            var cv = new CrossValidation(10, data, model_name, epochs, loss_method, optimizer, lr, batch_mode, batch_size, metrics, data_presentation);
            cv.Execute(model, data, label, seqs, times);



            // Training parameters WMSE

            epochs = 50;
            loss_method = ("weighted_square_error", RULWarning);
            optimizer = "rmsprop";
            lr = 0.01f;
            batch_mode = "random";
            batch_size = 1;
            metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", RULWarning) };
            data_presentation = "sequence";

            model_name = this.model_name + @"\WMSE\";


            // 10 fold Cross-Validation

            cv = new CrossValidation(10, data, model_name, epochs, loss_method, optimizer, lr, batch_mode, batch_size, metrics, data_presentation);
            cv.Execute(model, data, label, seqs, times);


        }

        public void AllVariablesRun()
        {

            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var RULWarning = 30;

            var (dataframe, labelframe) = DataRead.LoadData();

            var nomalization = new Zscore();

            nomalization.Fit(dataframe);

            dataframe = nomalization.Transform(dataframe);

            var (data, label, seqs, times) = DataRead.GetSequences(dataframe, labelframe);


            // Configuration model

            var model = new PredictionMLP();

            model.AddInputLayer(((int)data[0].shape[1]));
            model.AddDenseLayer(10, "sigmoid");
            model.AddDenseLayer(1, "linear");


            model.Verify();
            model.Summary();

            // Training parameters MSE

            var epochs = 50;
            (string name, int? param) loss_method = ("square_error", null);
            var optimizer = "rmsprop";
            var lr = 0.1f;
            var batch_mode = "continuos";
            var batch_size = 100;
            var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", RULWarning) };
            var data_presentation = "sample";

            var model_name = this.model_name + @"\MSE\";

            // 10 fold Cross-Validation

            var cv = new CrossValidation(10, data, model_name, epochs, loss_method, optimizer, lr, batch_mode, batch_size, metrics, data_presentation);
            cv.Execute(model, data, label, seqs, times);



            //// Training parameters WMSE

            //epochs = 100;
            //loss_method = ("weighted_square_error", RULWarning);
            //optimizer = "rmsprop";
            //lr = 0.01f;
            //batch_mode = "random";
            //batch_size = 1;
            //metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", RULWarning) };
            //data_presentation = "sequence";

            //model_name = this.model_name + @"\WMSE\";


            //// 10 fold Cross-Validation

            //cv = new CrossValidation(10, data, model_name, epochs, loss_method, optimizer, lr, batch_mode, batch_size, metrics, data_presentation);
            //cv.Execute(model, data, label, seqs, times);


        }

    }
}
