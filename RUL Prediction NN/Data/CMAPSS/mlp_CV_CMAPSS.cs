using RUL_Prediction_NN.Data;
using RUL_Prediction_NN.Model;
using RUL_Prediction_NN.Train;
using RUL_Prediction_NN.Transformation;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.KerasApi;

namespace RUL_Prediction_NN.CMAPSS
{
    class mlp_CV_CMAPSS
    {

        string directory;
        string model_name;

        public mlp_CV_CMAPSS(string model_name = "MLP CV CMAPSS")
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

            int? RULWarning = 30;

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


            // Configuration model

            var model = new PredictionMLP();

            model.AddInputLayer(((int)data[0].shape[1]));
            model.AddDenseLayer(8, "sigmoid");
            model.AddDenseLayer(1, "linear");



            //// Training parameters MSE

            //var epochs = 50;
            //(string name, int? param) loss_method = ("square_error", null);
            //var optimizer = "rmsprop";
            //var lr = 0.1f;
            //var batch_mode = "continuos";
            //var batch_size = 200;
            //var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", RULWarning) };
            //var data_presentation = "sample";

            //var model_name = this.model_name + @"\MSE\" ;

            //// 10 fold Cross-Validation

            //var cv = new CrossValidation(10, data, model_name, epochs, loss_method, optimizer, lr, batch_mode, batch_size, metrics, data_presentation);
            //cv.Execute(model, data, label, seqs, times);



            // Training parameters WMSE

            var epochs = 50;
            var loss_method = ("weighted_square_error", RULWarning);
            var optimizer = "rmsprop";
            var lr = 0.1f;
            var batch_mode = "random";
            var batch_size = 1;
            var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", RULWarning) };
            var data_presentation = "sequence";

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

            int? RULWarning = 30;

            var (dataframe, labelframe) = DataRead.LoadData();

            var variables = Enumerable.Range(5, 21);

            var normalization = new Zscore();

            normalization.Fit(dataframe);

            dataframe = normalization.Transform(dataframe);

            //var (data, label, seqs, times) = DataRead.GetSequences(dataframe, labelframe, variables);
            var (data, label, seqs, times) = DataRead.GetSequences(dataframe, labelframe, Enumerable.Range(5, 21));

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
            var batch_size = 200;
            var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", RULWarning) };
            var data_presentation = "sample";

            var model_name = this.model_name + @"\MSE\";

            // 10 fold Cross-Validation

            var cv = new CrossValidation(10, data, model_name, epochs, loss_method, optimizer, lr, batch_mode, batch_size, metrics, data_presentation);
            cv.Execute(model, data, label, seqs, times);


            // Training parameters WMSE

            //var epochs = 50;
            //var loss_method = ("weighted_square_error", RULWarning);
            //var optimizer = "rmsprop";
            //var lr = 0.01f;
            //var batch_mode = "random";
            //var batch_size = 1;
            //var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", RULWarning) };
            //var data_presentation = "sequence";

            //model_name = this.model_name + @"\WMSE\";


            //// 10 fold Cross-Validation

            //var cv = new CrossValidation(10, data, model_name, epochs, loss_method, optimizer, lr, batch_mode, batch_size, metrics, data_presentation);
            //cv.Execute(model, data, label, seqs, times);


        }

        public void AutoEncoderRun()
        {

            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            int? RULWarning = 30;

            var (dataframe, labelframe) = DataRead.LoadData();

            var variables = Enumerable.Range(5, 21);

            //var normalization = new Zscore();
            var normalization = new MinMax();

            normalization.Fit(dataframe);

            dataframe = normalization.Transform(dataframe);

            var (data, label, seqs, times) = DataRead.GetSequences(dataframe, labelframe, variables);


            var model_ae = new PredictionMLP();
            model_ae.AddInputLayer((Shape)(data[0].shape[1]));
            model_ae.AddDenseLayer(10, "relu");
            model_ae.AddDenseLayer((int)(data[0].shape[1]), "sigmoid");

            model_ae.Verify();

            model_ae.Summary();

            var x_train = np.concatenate(data.ToArray());

            model_ae.model.compile(keras.optimizers.RMSprop(0.001f), keras.losses.MeanSquaredError(), new string[] { "accuracy" });
            model_ae.model.fit(x_train, x_train, batch_size: 50, epochs: 50, validation_split: 0.1f);


            var X_feat = new List<NDArray>();

            foreach (var x in data)
            {
                X_feat.Add(model_ae.model.Layers[1].Apply(x)[0].numpy().reshape(model_ae.model.Layers[1].output_shape));
            }


            // Configuration model


            var model = new PredictionMLP();

            model.AddInputLayer(((int)X_feat[0].shape[1]));
            model.AddDenseLayer(6, "sigmoid");
            model.AddDenseLayer(1, "linear");

            model.Verify();
            model.Summary();

            //// Training parameters MSE

            //var epochs = 50;
            //(string name, int? param) loss_method = ("square_error", null);
            //var optimizer = "rmsprop";
            //var lr = 0.1f;
            //var batch_mode = "continuos";
            //var batch_size = 200;
            //var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", RULWarning) };
            //var data_presentation = "sample";

            //var model_name = this.model_name + @"\MSE\";

            //// 10 fold Cross-Validation

            //var cv = new CrossValidation(10, X_feat, model_name, epochs, loss_method, optimizer, lr, batch_mode, batch_size, metrics, data_presentation);
            //cv.Execute(model, X_feat, label, seqs, times);


            // Training parameters WMSE

            var epochs = 50;
            var loss_method = ("weighted_square_error", RULWarning);
            var optimizer = "rmsprop";
            var lr = 0.1f;
            var batch_mode = "random";
            var batch_size = 1;
            var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", RULWarning) };
            var data_presentation = "sequence";

            model_name = this.model_name + @"\WMSE\";


            // 10 fold Cross-Validation

            var cv = new CrossValidation(10, X_feat, model_name, epochs, loss_method, optimizer, lr, batch_mode, batch_size, metrics, data_presentation);
            cv.Execute(model, X_feat, label, seqs, times);


        }

    }
}
