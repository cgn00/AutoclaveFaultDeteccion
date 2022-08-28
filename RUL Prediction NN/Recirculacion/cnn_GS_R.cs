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
    public class cnn_GS_R
    {

        string directory;
        string model_name;

        public cnn_GS_R(string model_name = "CNN GS R")
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

            (data, label, seqs, times) = DataRead.SlideWindow(data, label, seqs, times, window_size);


            // CNN

            var epochs = new List<int> { 30 };
            (string loss, int? param) loss = ("square_error", null);
            var optimizer = "rmsprop";
            var learning_rate = new List<float> { 0.1f, 0.01f };
            var batch_mode = "continuos";
            var batch_size = new List<int> { 400 };
            var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", 120) };
            var nn_architecture = "CNN";

            var cnn_n_filters = new List<List<int>> { new List<int> { 10, 10, 10, 10, 1 } };
            var cnn_kernel_size = new List<List<Shape>> { new List<Shape> { (12, 2), (12, 2), (12, 2), (12, 2), (6, 2) } };
            var cnn_conv_activations = new List<List<string>> { new List<string> { "leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu" } };
            var cnn_dropout = new List<float> { 0.2f };
            var cnn_fc_n_hidden = new List<List<int>> { new List<int> { 10, 1 } };
            var cnn_fc_activations = new List<List<string>> { new List<string> { "leaky_relu", "linear" } };

            var X_train = new List<NDArray> { np.concatenate(data.ToArray()) };
            var Y_train = new List<NDArray> { np.concatenate(label.ToArray()) };
            var sequences = new List<NDArray> { np.concatenate(seqs.ToArray()) };
            var time = new List<NDArray> { np.concatenate(times.ToArray()) };

            var gs = new GridSearch(model_name, epochs, loss, optimizer, learning_rate, batch_mode, batch_size, metrics, nn_architecture, data, label, cnn_n_filters, cnn_kernel_size, cnn_conv_activations, cnn_dropout, cnn_fc_n_hidden, cnn_fc_activations);
            var (final_metrics, parameters) = gs.Execute(X_train, Y_train, sequences, time, "rmse");


        }

        public void AllVariablesRun()
        {

            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var window_size = 25;

            var (dataframe, labelframe) = DataRead.LoadData();

            var nomalization = new Zscore();

            nomalization.Fit(dataframe);

            dataframe = nomalization.Transform(dataframe);

            var (data, label, seqs, times) = DataRead.GetSequences(dataframe, labelframe);

            (data, label, seqs, times) = DataRead.SlideWindow(data, label, seqs, times, window_size);


            // Training parameters

            var epochs = new List<int> { 10 };
            (string loss, int? param) loss = ("square_error", null);
            var optimizer = "adam";
            var learning_rate = new List<float> { 0.01f };
            var batch_mode = "continuos";
            var batch_size = new List<int> { 400 };
            var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", 120) };
            var nn_architecture = "CNN";


            // CNN      

            var cnn_n_filters = new List<List<int>> { new List<int> { 5, 5, 1 } };
            var cnn_kernel_size = new List<List<Shape>> { new List<Shape> { (12, 1), (12, 1), (6, 1) } };
            var cnn_conv_activations = new List<List<string>> { new List<string> { "leaky_relu", "leaky_relu", "leaky_relu" } };
            var cnn_dropout = new List<float> { 0.5f };
            var cnn_fc_n_hidden = new List<List<int>> { new List<int> { 10, 1 } };
            var cnn_fc_activations = new List<List<string>> { new List<string> { "leaky_relu", "linear" } };
            var cnn_pooling = new List<Shape>();

            var X_train = new List<NDArray> { np.concatenate(data.ToArray()) };
            var Y_train = new List<NDArray> { np.concatenate(label.ToArray()) };
            var sequences = new List<NDArray> { np.concatenate(seqs.ToArray()) };
            var time = new List<NDArray> { np.concatenate(times.ToArray()) };

            var gs = new GridSearch(model_name, epochs, loss, optimizer, learning_rate, batch_mode, batch_size, metrics, nn_architecture, data, label, cnn_n_filters, cnn_kernel_size, cnn_conv_activations, cnn_dropout, cnn_fc_n_hidden, cnn_fc_activations, cnn_pooling);
            var (final_metrics, parameters) = gs.Execute(X_train, Y_train, sequences, time, "rmse");

            //var ind = final_metrics.IndexOf(final_metrics.Min());
            //var best = parameters[ind];


        }

        public void AllVariablesRun1()
        {

            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var window_size = 25;

            var (dataframe, labelframe) = DataRead.LoadData();

            var nomalization = new Zscore();

            nomalization.Fit(dataframe);

            dataframe = nomalization.Transform(dataframe);

            var (data, label, seqs, times) = DataRead.GetSequences(dataframe, labelframe);

            (data, label, seqs, times) = DataRead.SlideWindow(data, label, seqs, times, window_size);


            // Training parameters

            var epochs = new List<int> { 30 };
            (string loss, int? param) loss = ("square_error", null);
            var optimizer = "rmsprop";
            var learning_rate = new List<float> { 0.01f, 0.001f };
            var batch_mode = "continuos";
            var batch_size = new List<int> { 400 };
            var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", 120) };
            var nn_architecture = "CNN";


            // CNN      

            var cnn_n_filters = new List<List<int>> { new List<int> { 5, 1 } };
            var cnn_kernel_size = new List<List<Shape>> { new List<Shape> { (12, 1), (12, 1) } };
            var cnn_conv_activations = new List<List<string>> { new List<string> { "leaky_relu", "leaky_relu" } };

            var cnn_dropout = new List<float> { 0.5f };
            var cnn_fc_n_hidden = new List<List<int>> { new List<int> { 10, 1 } };
            var cnn_fc_activations = new List<List<string>> { new List<string> { "leaky_relu", "linear" } };
            var cnn_pooling = new List<Shape> { (1, 3), (3, 3), (3, 1) };


            var X_train = new List<NDArray> { np.concatenate(data.ToArray()) };
            var Y_train = new List<NDArray> { np.concatenate(label.ToArray()) };
            var sequences = new List<NDArray> { np.concatenate(seqs.ToArray()) };
            var time = new List<NDArray> { np.concatenate(times.ToArray()) };

            var gs = new GridSearch(model_name, epochs, loss, optimizer, learning_rate, batch_mode, batch_size, metrics, nn_architecture, data, label, cnn_n_filters, cnn_kernel_size, cnn_conv_activations, cnn_dropout, cnn_fc_n_hidden, cnn_fc_activations, cnn_pooling);
            var (final_metrics, parameters) = gs.Execute(X_train, Y_train, sequences, time, "rmse");



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

            //var (data, label, _, _) = DataRead.GetSequences(dataframe, labelframe, Enumerable.Range(5, 21));
            var (data, label, seqs, times) = DataRead.GetSequences(dataframe, labelframe);

            var ae_n_hidden = new List<int> { 2, 8, 10, 12 };
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
                model_ae.AddDenseLayer(n_h, "leaky_relu");
                model_ae.AddDenseLayer(n_output, "leaky_relu");

                //model_ae.Verify();

                model_ae.Summary();


                var x_train = np.concatenate(data.ToArray());


                if (!(File.Exists(new_directory + "ae_final_weigths.hdf5")))
                {

                    model_ae.model.save_weights(new_directory + "ae_initial_weigths.hdf5");

                    model_ae.model.compile(keras.optimizers.RMSprop(0.1f), keras.losses.MeanSquaredError(), new string[] { "accuracy" });
                    model_ae.model.fit(x_train, x_train, batch_size: 400, epochs: 50, validation_split: 0.1f);

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


                // Training parameters

                var epochs = new List<int> { 30 };
                (string loss, int? param) loss = ("square_error", null);
                var optimizer = "rmsprop";
                var learning_rate = new List<float> { 0.01f, 0.001f };
                var batch_mode = "continuos";
                var batch_size = new List<int> { 400 };
                var metrics = new List<(string name, int? param)> { ("rmse", null), ("rwmse", 120) };
                var nn_architecture = "CNN";


                // CNN      

                var cnn_n_filters = new List<List<int>> { new List<int> { 5, 5, 1 }, new List<int> { 10, 10, 1 }, new List<int> { 5, 5, 1 }, new List<int> { 10, 10, 1 } };
                var cnn_kernel_size = new List<List<Shape>> { new List<Shape> { (12, 1), (12, 1), (6, 1) }, new List<Shape> { (12, 1), (12, 1), (6, 1) }, new List<Shape> { (12, 1), (12, 1), (6, 1) }, new List<Shape> { (12, 1), (12, 1), (6, 1) } };
                var cnn_conv_activations = new List<List<string>> { new List<string> { "leaky_relu", "leaky_relu", "leaky_relu" }, new List<string> { "leaky_relu", "leaky_relu", "leaky_relu" }, new List<string> { "relu", "relu", "relu" }, new List<string> { "relu", "relu", "relu" } };
                var cnn_dropout = new List<float> { 0.2f, 0.5f };
                var cnn_fc_n_hidden = new List<List<int>> { new List<int> { 10, 1 } };
                var cnn_fc_activations = new List<List<string>> { new List<string> { "sigmoid", "linear" } };


                var X_train = new List<NDArray> { np.concatenate(data.ToArray()) };
                var Y_train = new List<NDArray> { np.concatenate(label.ToArray()) };
                var sequences = new List<NDArray> { np.concatenate(seqs.ToArray()) };
                var time = new List<NDArray> { np.concatenate(times.ToArray()) };


                var gs = new GridSearch(new_model_name, epochs, loss, optimizer, learning_rate, batch_mode, batch_size, metrics, nn_architecture, X_feat, label, cnn_n_filters, cnn_kernel_size, cnn_conv_activations, cnn_dropout, cnn_fc_n_hidden, cnn_fc_activations);
                var (final_metrics, parameters) = gs.Execute(X_train, Y_train, sequences, time, "rmse");

            }


        }


    }
}
