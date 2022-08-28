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

namespace RUL_Prediction_NN.Recirculacion
{
    public class mlp_GS_R
    {

        string directory;
        string model_name;

        public mlp_GS_R(string model_name = "MLP GS R")
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

            var window_size = 25;

            var (dataframe, labelframe) = DataRead.LoadData();

            var variability = new Variability();

            var wp = 1.0;
            var wm = 1.0;
            var wt = 0.0;
            var theresold = 1.0;
            var len = 100;

            variability.Fit(directory, wp, wm, wt, theresold, 10, len);

            dataframe = variability.Transform(dataframe);

            var nomalization = new Zscore();

            nomalization.Fit(dataframe);

            dataframe = nomalization.Transform(dataframe);

            var (data, label, seqs, times) = DataRead.GetSequences(dataframe, labelframe);

            // MLP

            var epochs = new List<int> { 30 };
            (string loss, int? param) loss = ("weighted_square_error", 120);
            var optimizer = "rmsprop";
            var learning_rate = new List<float> { 0.2f, 0.1f, 0.01f };
            var batch_mode = "random";
            var batch_size = new List<int> { 1 };
            var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", 120) };
            var nn_architecture = "MLP";

            var n_hidden = new List<List<int>> { new List<int> { 4, 1 }, new List<int> { 8, 1 }, new List<int> { 10, 1 } };
            var n_activations = new List<List<string>> { new List<string> { "relu", "linear" }, new List<string> { "relu", "linear" }, new List<string> { "relu", "linear" } };

            var X_train = new List<NDArray> { np.concatenate(data.ToArray()) };
            var Y_train = new List<NDArray> { np.concatenate(label.ToArray()) };
            var sequences = new List<NDArray> { np.concatenate(seqs.ToArray()) };
            var time = new List<NDArray> { np.concatenate(times.ToArray()) };

            var gs = new GridSearch(model_name, epochs, loss, optimizer, learning_rate, batch_mode, batch_size, metrics, nn_architecture, data, label, n_hidden, n_activations, false);
            var (final_metrics, parameters) = gs.Execute(X_train, Y_train, sequences, time, "rmse");

            //var ind = final_metrics.IndexOf(final_metrics.Min());
            //var best = parameters[ind];


        }

        public void AllVariablesRun()
        {


            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var (dataframe, labelframe) = DataRead.LoadData();

            var nomalization = new Zscore();

            nomalization.Fit(dataframe);

            dataframe = nomalization.Transform(dataframe);

            var (data, label, seqs, times) = DataRead.GetSequences(dataframe, labelframe);

            // MLP

            var epochs = new List<int> { 10 };
            (string loss, int? param) loss = ("square_error", null);
            //(string loss, int? param) loss = ("weighted_square_error", 120);
            var optimizer = "rmsprop";
            var learning_rate = new List<float> { 0.1f };
            var batch_mode = "continuos";
            //var batch_mode = "random";
            var batch_size = new List<int> { 400 };
            //var batch_size = new List<int> { 1 };
            var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", 120) };
            var nn_architecture = "MLP";

            var n_hidden = new List<List<int>> { new List<int> { 10, 1 } };
            var n_activations = new List<List<string>> {  new List<string> { "sigmoid", "linear" } };

            var X_train = new List<NDArray> { np.concatenate(data.ToArray()) };
            var Y_train = new List<NDArray> { np.concatenate(label.ToArray()) };
            var sequences = new List<NDArray> { np.concatenate(seqs.ToArray()) };
            var time = new List<NDArray> { np.concatenate(times.ToArray()) };

            //var X_train = data;
            //var Y_train = label;
            //var sequences = seqs;
            //var time = times;

            var gs = new GridSearch(model_name, epochs, loss, optimizer, learning_rate, batch_mode, batch_size, metrics, nn_architecture, data, label, n_hidden, n_activations);
            var (final_metrics, parameters) = gs.Execute(X_train, Y_train, sequences, time, "rmse");

            //var ind = final_metrics.IndexOf(final_metrics.Min());
            //var best = parameters[ind];


        }

        public void AutoEncoderRun()
        {

            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var (dataframe, labelframe) = DataRead.LoadData();

            //var nomalization = new Zscore();
            var nomalization = new MinMax();

            nomalization.Fit(dataframe);

            dataframe = nomalization.Transform(dataframe);

            var (data, label, seqs, times) = DataRead.GetSequences(dataframe, labelframe);

            var ae_n_hidden = new List<int> { 6, 8, 10 };
            var n_input = data[0].shape[1];
            var n_output = (int)data[0].shape[1];


            foreach (var n_h in ae_n_hidden)
            {

                var new_directory = directory + @"\ae_" + n_h + @"\";

                if (!Directory.Exists(new_directory))
                {
                    Directory.CreateDirectory(new_directory);
                }


                var model_ae = new PredictionMLP();
                model_ae.AddInputLayer((Shape)(n_input));
                model_ae.AddDenseLayer(n_h, "relu");
                model_ae.AddDenseLayer(n_output, "sigmoid");

                model_ae.Verify();

                model_ae.Summary();


                var x_train = np.concatenate(data.ToArray());


                if (!(File.Exists(new_directory + "ae_final_weigths.hdf5")))
                {

                    model_ae.model.save_weights(new_directory + "ae_initial_weigths.hdf5");

                    model_ae.model.compile(keras.optimizers.RMSprop(0.001f), keras.losses.MeanSquaredError(), new string[] { "accuracy" });
                    model_ae.model.fit(x_train, x_train, batch_size: 100, epochs: 50, validation_split: 0.1f);

                    model_ae.model.save_weights(new_directory + "ae_final_weigths.hdf5");
                }

                else
                {
                    model_ae.model.load_weights(new_directory + "ae_final_weigths.hdf5");
                }

                var new_model_name = model_name + @"\ae_" + n_h;


                var X_feat = new List<NDArray>();

                foreach (var x in data)
                {
                    X_feat.Add(model_ae.model.Layers[1].Apply(x)[0].numpy().reshape(model_ae.model.Layers[1].output_shape));
                }


                // MLP

                var epochs = new List<int> { 50 };
                (string loss, int? param) loss = ("square_error", null);
                var optimizer = "rmsprop";
                var learning_rate = new List<float> { 0.2f, 0.1f, 0.01f };
                var batch_mode = "continuos";
                var batch_size = new List<int> { 400 };
                var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", 120) };
                var nn_architecture = "MLP";

                var n_hidden = new List<List<int>> { new List<int> { 4, 1 } };
                var n_activations = new List<List<string>> { new List<string> { "sigmoid", "linear" } };

                var X_train = new List<NDArray> { np.concatenate(data.ToArray()) };
                var Y_train = new List<NDArray> { np.concatenate(label.ToArray()) };
                var sequences = new List<NDArray> { np.concatenate(seqs.ToArray()) };
                var time = new List<NDArray> { np.concatenate(times.ToArray()) };

                var gs = new GridSearch(new_model_name, epochs, loss, optimizer, learning_rate, batch_mode, batch_size, metrics, nn_architecture, X_feat, label, n_hidden, n_activations, false);
                var (final_metrics, parameters) = gs.Execute(X_train, Y_train, sequences, time, "rmse");


            }


        }


    }
}
