using RUL_Prediction_NN.Data;
using RUL_Prediction_NN.Misc;
using RUL_Prediction_NN.Model;
using RUL_Prediction_NN.Train;
using RUL_Prediction_NN.Transformation;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace RUL_Prediction_NN.CMAPSS
{
    public class cnn_CV_CMAPSS
    {

        string directory;
        string model_name;

        public cnn_CV_CMAPSS(string model_name = "CNN CV CMAPSS")
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
            var theresold = 1.5;
            var window_size = 30;
            int len = 100;

            variability.Fit(directory, wp, wm, wt, theresold, window_size, len);

            dataframe = variability.Transform(dataframe);

            var nomalization = new Zscore();

            nomalization.Fit(dataframe);

            dataframe = nomalization.Transform(dataframe);

            var (data, label, seqs, times) = DataRead.GetSequences(dataframe, labelframe);

            (data, label, seqs, times) = DataRead.SlideWindow(data, label, seqs, times, window_size);


            // Configuration model

            var model = new PredictionCNN();

            model.AddInputLayer((window_size, (int)data[0].shape[2], 1));
            model.AddConvLayer(10, (10, 1), "leaky_relu");
            model.AddConvLayer(10, (10, 1), "leaky_relu");
            model.AddConvLayer(10, (10, 1), "leaky_relu");
            model.AddConvLayer(10, (10, 1), "leaky_relu");
            model.AddConvLayer(1, (3, 1), "leaky_relu");
            model.AddFlattenLayer();
            model.AddDropoutLayer(0.5f);
            model.AddDenseLayer(10, "leaky_relu");
            model.AddDenseLayer(1, "linear");

            model.Verify();
            model.Summary();


            // Training parameters MSE

            var epochs = 30;
            (string name, int? param) loss_method = ("square_error", null);
            var optimizer = "rmsprop";
            var lr = 0.001f;
            var batch_mode = "continuos";
            var batch_size = 200;
            var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", RULWarning) };
            var data_presentation = "sample";

            var model_name = this.model_name + @"\MSE\";

            // 10 fold Cross-Validation

            //var cv = new CrossValidation(10, data, model_name, epochs, loss_method, optimizer, lr, batch_mode, batch_size, metrics, data_presentation);
            //cv.Execute(model, data, label, seqs, times);


            // Training parameters WMSE

            epochs = 30;
            loss_method = ("weighted_square_error", RULWarning);
            optimizer = "rmsprop";
            lr = 0.001f;
            batch_mode = "random";
            batch_size = 1;
            metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", RULWarning) };
            data_presentation = "sequence";

            model_name = this.model_name + @"\WMSE\";

            // 10 fold Cross-Validation

            var cv = new CrossValidation(10, data, model_name, epochs, loss_method, optimizer, lr, batch_mode, batch_size, metrics, data_presentation);
            cv.Execute(model, data, label, seqs, times);

        }

        public void AllVariablesRun()
        {


            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var RULWarning = 30;
            var window_size = 30;

            var (dataframe, labelframe) = DataRead.LoadData();

            var nomalization = new Zscore();

            nomalization.Fit(dataframe);

            dataframe = nomalization.Transform(dataframe);

            var (data, label, seqs, times) = DataRead.GetSequences(dataframe, labelframe, Enumerable.Range(5, 21));
            //var (data, label, seqs, times) = DataRead.GetSequences(dataframe, labelframe);

            (data, label, seqs, times) = DataRead.SlideWindow(data, label, seqs, times, window_size);


            // Configuration model

            var model = new PredictionCNN();

            model.AddInputLayer((window_size, (int)data[0].shape[2], 1));
            model.AddConvLayer(10, (10, 1), "tanh");
            model.AddConvLayer(10, (10, 1), "tanh");
            model.AddConvLayer(10, (10, 1), "tanh");
            model.AddConvLayer(10, (10, 1), "tanh");
            model.AddConvLayer(1, (3, 1), "tanh");
            model.AddFlattenLayer();
            model.AddDropoutLayer(0.5f);
            model.AddDenseLayer(10, "sigmoid");
            model.AddDenseLayer(1, "linear");

            model.Verify();
            model.Summary();


            // Training parameters MSE

            var epochs = 250;
            (string name, int? param) loss_method = ("square_error", null);
            var optimizer = "rmsprop";
            var lr = 0.001f;
            var batch_mode = "continuos";
            var batch_size = 512;
            var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", RULWarning) };
            var data_presentation = "sample";

            var model_name = this.model_name + @"\MSE\";

            // 10 fold Cross-Validation

            var cv = new CrossValidation(10, data, model_name, epochs, loss_method, optimizer, lr, batch_mode, batch_size, metrics, data_presentation);
            cv.Execute(model, data, label, seqs, times);


            // Training parameters WMSE

            epochs = 250;
            loss_method = ("weighted_square_error", RULWarning);
            optimizer = "rmsprop";
            lr = 0.001f;
            batch_mode = "random";
            batch_size = 5;
            metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", RULWarning) };
            data_presentation = "sequence";

            model_name = this.model_name + @"\WMSE\";

            // 10 fold Cross-Validation

            cv = new CrossValidation(10, data, model_name, epochs, loss_method, optimizer, lr, batch_mode, batch_size, metrics, data_presentation);
            cv.Execute(model, data, label, seqs, times);

        }

    }
}
