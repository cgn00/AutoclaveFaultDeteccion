using Accord.Math;
using RUL_Prediction_NN.Misc;
using RUL_Prediction_NN.Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Optimizers;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace RUL_Prediction_NN.Train
{
    public class GridSearch
    {


        string save_directory;

        List<NDArray> validation_x, validation_y;

        string optimizer;
        List<float> learning_rate;


        List<int> epochs;
        List<int> batch_size;
        List<(string name, int? param)> metrics;


        string batch_mode;
        (string loss, int? param) loss_method;

        bool batch_normalization = false;


        string model_name;
        string nn_architecture;


        // mlp architecture
        List<List<int>> mlp_n_hidden;
        List<List<string>> mlp_activation_hidden;


        // cnn architecture
        List<List<int>> cnn_n_filters;
        List<List<Shape>> cnn_kernel_size;
        List<List<string>> cnn_conv_activations;
        List<List<int>> cnn_fc_n_hidden;
        List<List<string>> cnn_fc_activations;
        List<float> cnn_dropout;
        List<Shape> cnn_pooling;


        // Comparison metrics
        List<float> final_metrics;
        List<object[]> parameters;


        public GridSearch(string grid_name, params object[] parameters)
        {

            epochs = (List<int>)parameters[0];
            loss_method = ((string loss, int? param))parameters[1];
            optimizer = (string)parameters[2];
            learning_rate = (List<float>)parameters[3];
            batch_mode = (string)parameters[4];
            batch_size = (List<int>)parameters[5];
            metrics = (List<(string name, int? param)>)parameters[6];
            nn_architecture = (string)parameters[7];
            validation_x = (List<NDArray>)parameters[8];
            validation_y = (List<NDArray>)parameters[9];


            if (nn_architecture == "MLP")
            {
                mlp_n_hidden = (List<List<int>>)parameters[10];
                mlp_activation_hidden = (List<List<string>>)parameters[11];
                //batch_normalization = (bool)parameters[12];
            }

            else if (nn_architecture == "CNN")
            {
                cnn_n_filters = (List<List<int>>)parameters[10];
                cnn_kernel_size = (List<List<Shape>>)parameters[11];
                cnn_conv_activations = (List<List<string>>)parameters[12];
                cnn_dropout = (List<float>)parameters[13];
                cnn_fc_n_hidden = (List<List<int>>)parameters[14];
                cnn_fc_activations = (List<List<string>>)parameters[15];
                cnn_pooling = (List<Shape>)parameters[16];
            }

            else
            {
                throw new Exception(message: "");
            }

            save_directory = Directory.GetCurrentDirectory() + @"\results\" + grid_name + @"\";

            final_metrics = new List<float>();
            this.parameters = new List<object[]>();

        }

        public (List<float>, List<object[]>) Execute(List<NDArray> X, List<NDArray> Y, List<NDArray> seqs, List<NDArray> times, string comparison_metric)
        {


            foreach (var e in epochs)
            {

                foreach (var b_s in batch_size)
                {

                    foreach (var l_r in learning_rate)
                    {

                        if (nn_architecture == "MLP")
                        {

                            foreach (var h in mlp_n_hidden.Zip(mlp_activation_hidden))
                            {

                                var n_input = (int)X[0].shape[1];
                                var input_shape = (n_input);

                                model_name = save_directory + nn_architecture + "_" + optimizer + "_";

                                if (loss_method.loss == "square_error")
                                {
                                    model_name = model_name + "sq_e";
                                }

                                else if (loss_method.loss == "weighted_square_error")
                                {
                                    model_name = model_name + "wsq_e";
                                }

                                model_name = model_name + "_" + batch_mode + "_" + b_s +
                                              "_" + l_r + "_" + e + "_";

                                foreach (var a in h.First.Zip(h.Second))
                                {
                                    model_name = model_name + "_" + a.First + "_" + a.Second;
                                }

                                var mlp = new PredictionMLP();

                                mlp.AddInputLayer(input_shape);

                                for (int i = 0; i < h.First.Count; i++)
                                {
                                    mlp.AddDenseLayer(h.First[i], h.Second[i], batch_normalization);
                                }

                                mlp.Verify();

                                mlp.Summary();


                                var (_, metric_training) = PredictingTraining.Train(model: mlp, data: X, label: Y,
                                                                 model_name,
                                                                 e,
                                                                 loss_method,
                                                                 optimizer,
                                                                 l_r,
                                                                 batch_mode,
                                                                 b_s,
                                                                 metrics,
                                                                 validation_x,
                                                                 validation_y);

                                var temp_preds = new List<NDArray>();

                                foreach (var a in X)
                                {
                                    temp_preds.Add(mlp.model.Apply(a.reshape(mlp.model.Layers[0].output_shape))[0].numpy());
                                }

                                var partition_id = (np.ones((np.concatenate(X.ToArray()).shape[0], 1), tf.float32) * (0 + 1)).reshape((-1, 1));
                                var preds = np.concatenate(temp_preds.ToArray());
                                var trues = np.concatenate(Y.ToArray());
                                var sequences = np.concatenate(seqs.ToArray());
                                var time = np.concatenate(times.ToArray());

                                //var sequences = (np.ones((np.concatenate(X.ToArray()).shape[0], 1), tf.float32) * (0 + 1)).reshape((-1, 1));
                                //var sequences = np.linspace<int>(1, 100, 100);
                                //var time = (np.ones((np.concatenate(X.ToArray()).shape[0], 1), tf.float32) * (0 + 1)).reshape((-1, 1));

                                tool.save_model_output_prediction(model_name + @"\prognostics\" + @"\model_output.csv", trues, preds, sequences, time, partition_id);

                                model_name = null;

                                foreach (var m in metric_training)
                                {
                                    if (m.metric == comparison_metric)
                                    {
                                        final_metrics.Add(m.value);
                                    }
                                }

                                parameters.Add(new object[] { e, b_s, l_r, h.First, h.Second });

                            }


                        }


                        if (nn_architecture == "CNN")
                        {

                            var list = cnn_n_filters.Zip(cnn_kernel_size, (e1, e2) => new { e1, e2 }).Zip(cnn_conv_activations, (z1, e3) => Tuple.Create(z1.e1, z1.e2, e3));

                            foreach (var f in list)
                            {

                                foreach (var dr in cnn_dropout)
                                {

                                    if (cnn_pooling.Count != 0)
                                    {


                                        foreach (var p in cnn_pooling)
                                        {

                                            foreach (var fc_h in cnn_fc_n_hidden.Zip(cnn_fc_activations))
                                            {

                                                var n_input = (int)X[0].shape[2];
                                                var window_size = (int)X[0].shape[1];
                                                var input_shape = (window_size, n_input, 1);

                                                model_name = save_directory + nn_architecture + "_" + optimizer + "_";

                                                if (loss_method.loss == "square_error")
                                                {
                                                    model_name = model_name + "sqe_";
                                                }

                                                else if (loss_method.loss == "weighted_square_error")
                                                {
                                                    model_name = model_name + "wsqe_";
                                                }

                                                if (batch_mode == "random")
                                                {
                                                    model_name = model_name + "rand";
                                                }

                                                else if (batch_mode == "continuos")
                                                {
                                                    model_name = model_name + "cont";
                                                }

                                                model_name = model_name + "_" + b_s +
                                                             "_" + l_r + "_" + e + "_" + window_size + "_" + dr;

                                                model_name = model_name + "_" + f.Item1.Zip(f.Item2).ToList()[0].First + "_" + f.Item1.Zip(f.Item2).ToList()[0].Second;

                                                model_name = model_name + "_pooling";

                                                foreach (var a in fc_h.First.Zip(fc_h.Second))
                                                {
                                                    if (a.Second == "sigmoid")
                                                    {
                                                        model_name = model_name + "_" + a.First + "_" + "sigm";
                                                    }

                                                    else if (a.Second == "leaky_relu")
                                                    {
                                                        model_name = model_name + "_" + a.First + "_" + "lk";
                                                    }

                                                    else
                                                    {
                                                        model_name = model_name + "_" + a.First + "_" + a.Second;
                                                    }

                                                }

                                                var cnn = new PredictionCNN();

                                                cnn.AddInputLayer(input_shape);

                                                for (int i = 0; i < f.Item1.Count; i++)
                                                {
                                                    if (i != f.Item1.Count - 1)
                                                    {
                                                        cnn.AddConvLayer(f.Item1[i], f.Item2[i], f.Item3[i]);
                                                        cnn.AddPoolingLayer(p);
                                                    }

                                                    else
                                                    {
                                                        cnn.AddConvLayer(f.Item1[i], f.Item2[i], f.Item3[i]);
                                                    }

                                                }

                                                cnn.AddFlattenLayer();
                                                cnn.AddDropoutLayer(dr);

                                                for (int i = 0; i < fc_h.First.Count; i++)
                                                {
                                                    cnn.AddDenseLayer(fc_h.First[i], fc_h.Second[i]);
                                                }

                                                cnn.Verify();
                                                cnn.Summary();


                                                


                                                var (_, metric_training) = PredictingTraining.Train(model: cnn, data: X, label: Y,
                                                                         model_name,
                                                                         e,
                                                                         loss_method,
                                                                         optimizer,
                                                                         l_r,
                                                                         batch_mode,
                                                                         b_s,
                                                                         metrics,
                                                                         validation_x,
                                                                         validation_y);


                                                var temp_preds = new List<NDArray>();

                                                foreach (var a in X)
                                                {
                                                    temp_preds.Add(cnn.model.Apply(a.reshape(cnn.model.Layers[0].output_shape))[0].numpy());
                                                }

                                                var partition_id = (np.ones((np.concatenate(X.ToArray()).shape[0], 1), tf.float32) * (0 + 1)).reshape((-1, 1));
                                                var preds = np.concatenate(temp_preds.ToArray());
                                                var trues = np.concatenate(Y.ToArray());
                                                var sequences = np.concatenate(seqs.ToArray());
                                                var time = np.concatenate(times.ToArray());

                                                //var sequences = (np.ones((np.concatenate(X.ToArray()).shape[0], 1), tf.float32) * (0 + 1)).reshape((-1, 1));
                                                //var time = (np.ones((np.concatenate(X.ToArray()).shape[0], 1), tf.float32) * (0 + 1)).reshape((-1, 1));

                                                tool.save_model_output_prediction(model_name + @"\prognostics\" + @"\model_output.csv", trues, preds, sequences, time, partition_id);

                                                model_name = null;

                                                foreach (var m in metric_training)
                                                {
                                                    if (m.metric == comparison_metric)
                                                    {
                                                        final_metrics.Add(m.value);
                                                    }
                                                }

                                                parameters.Add(new object[] { e, b_s, l_r, f.Item1, f.Item2, f.Item3, dr, fc_h.First, fc_h.Second });

                                            }

                                        }

                                    }

                                    else
                                    {
                                        foreach (var fc_h in cnn_fc_n_hidden.Zip(cnn_fc_activations))
                                        {

                                            var n_input = (int)X[0].shape[2];
                                            var window_size = (int)X[0].shape[1];
                                            var input_shape = (window_size, n_input, 1);

                                            model_name = save_directory + nn_architecture + "_" + optimizer + "_";

                                            if (loss_method.loss == "square_error")
                                            {
                                                model_name = model_name + "sqe_";
                                            }

                                            else if (loss_method.loss == "weighted_square_error")
                                            {
                                                model_name = model_name + "wsqe_";
                                            }

                                            if (batch_mode == "random")
                                            {
                                                model_name = model_name + "rand";
                                            }

                                            else if (batch_mode == "continuos")
                                            {
                                                model_name = model_name + "cont";
                                            }

                                            model_name = model_name + "_" + b_s +
                                                         "_" + l_r + "_" + e + "_" + window_size + "_" + dr;

                                            model_name = model_name + "_" + f.Item1.Zip(f.Item2).ToList()[0].First + "_" + f.Item1.Zip(f.Item2).ToList()[0].Second;

                                            foreach (var a in fc_h.First.Zip(fc_h.Second))
                                            {
                                                if (a.Second == "sigmoid")
                                                {
                                                    model_name = model_name + "_" + a.First + "_" + "sigm";
                                                }

                                                else if (a.Second == "leaky_relu")
                                                {
                                                    model_name = model_name + "_" + a.First + "_" + "lk";
                                                }

                                                else
                                                {
                                                    model_name = model_name + "_" + a.First + "_" + a.Second;
                                                }

                                            }

                                            var cnn = new PredictionCNN();

                                            cnn.AddInputLayer(input_shape);

                                            for (int i = 0; i < f.Item1.Count; i++)
                                            {
                                                cnn.AddConvLayer(f.Item1[i], f.Item2[i], f.Item3[i]);
                                            }

                                            cnn.AddFlattenLayer();
                                            cnn.AddDropoutLayer(dr);

                                            for (int i = 0; i < fc_h.First.Count; i++)
                                            {
                                                cnn.AddDenseLayer(fc_h.First[i], fc_h.Second[i]);
                                            }

                                            cnn.Verify();
                                            cnn.Summary();

                                            // QUITAR
                                            model_name = save_directory + nn_architecture;

                                            var (_, metric_training) = PredictingTraining.Train(model: cnn, data: X, label: Y,
                                                                     model_name,
                                                                     e,
                                                                     loss_method,
                                                                     optimizer,
                                                                     l_r,
                                                                     batch_mode,
                                                                     b_s,
                                                                     metrics,
                                                                     validation_x,
                                                                     validation_y);


                                            var temp_preds = new List<NDArray>();

                                            foreach (var a in X)
                                            {
                                                temp_preds.Add(cnn.model.Apply(a.reshape(cnn.model.Layers[0].output_shape))[0].numpy());
                                            }

                                            var partition_id = (np.ones((np.concatenate(X.ToArray()).shape[0], 1), tf.float32) * (0 + 1)).reshape((-1, 1));
                                            var preds = np.concatenate(temp_preds.ToArray());
                                            var trues = np.concatenate(Y.ToArray());
                                            //var sequences = (np.ones((np.concatenate(X.ToArray()).shape[0], 1), tf.float32) * (0 + 1)).reshape((-1, 1));
                                            //var time = (np.ones((np.concatenate(X.ToArray()).shape[0], 1), tf.float32) * (0 + 1)).reshape((-1, 1));
                                            var sequences = np.concatenate(seqs.ToArray());
                                            var time = np.concatenate(times.ToArray());


                                            tool.save_model_output_prediction(model_name + @"\prognostics\" + @"\model_output.csv", trues, preds, sequences, time, partition_id);

                                            model_name = null;

                                            foreach (var m in metric_training)
                                            {
                                                if (m.metric == comparison_metric)
                                                {
                                                    final_metrics.Add(m.value);
                                                }
                                            }

                                            parameters.Add(new object[] { e, b_s, l_r, f.Item1, f.Item2, f.Item3, dr, fc_h.First, fc_h.Second });

                                        }
                                    }


                                }


                            }


                        }


                    }

                }


            }


            return (final_metrics, parameters);

        }



    }

}
