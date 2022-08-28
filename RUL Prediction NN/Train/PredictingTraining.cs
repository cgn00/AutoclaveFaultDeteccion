using Accord.Math;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
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
using RUL_Prediction_NN.Model;
using RUL_Prediction_NN.Data;
using RUL_Prediction_NN.Misc;

namespace RUL_Prediction_NN.Train
{
    public static class PredictingTraining
    {

        // Save path
        static string save_directory;

        // Data train
        static List<NDArray> X, Y;

        // Validation data
        static List<NDArray> validation_x_sequence, validation_y_sequence;

        // Training objects
        static OptimizerV2 optimizer;
        static Loss_Delegate loss;
        static IPredictionModel prediction_model;

        // Training Parameters
        static int epochs;

        static float learning_rate;

        // Metrics
        static List<(string name, int? param, Metric_Delegate method)> metrics;
        static List<(string metric, float value)> final_metrics_value;

        // Batch
        static Batch_Delegate batch;
        static int batch_size;

        static int? RULwarning;




        public static (IPredictionModel, List<(string metric, float value)>) Train(IPredictionModel model, List<NDArray> data, List<NDArray> label, params object[] parameters)
        {

            /*
             *  paramters[0] = model name
             *  paramters[1] = epochs
             *  paramters[2] = loss
             *  paramters[3] = optimizer
             *  paramters[4] = lr
             *  paramters[5] = batch mode
             *  paramters[6] = batch size
             *  paramters[7] = metrics
             *  paramters[8] = validation x seq
             *  paramters[9] = validation y seq
             */


            var model_name = (string)parameters[0];
            epochs = (int)parameters[1];
            var loss_method = ((string loss, int? param))parameters[2];
            var optimizer = (string)parameters[3];
            var lr = (float)parameters[4];
            var batch_mode = (string)parameters[5];
            batch_size = (int)parameters[6];
            var metrics = (List<(string name, int? param)>)parameters[7];
            validation_x_sequence = (List<NDArray>)parameters[8];
            validation_y_sequence = (List<NDArray>)parameters[9];


            // Name of directory
            save_directory = model_name + @"\prognostics\";

            // Sequential model
            prediction_model = model;

            // Data and labels
            X = data;
            Y = label;


            // Loss function            
            if (loss_method.loss == "square_error")
            {
                loss = prognostics.square_error;
                RULwarning = null;
            }

            else if (loss_method.loss == "weighted_square_error")
            {
                loss = prognostics.weighted_square_error;
                RULwarning = loss_method.param;
            }

            else
            {
                throw new Exception(message: "Invalid loss method");
            }


            learning_rate = lr;

            // Training
            if (optimizer == "rmsprop")
            {
                //PredictingTraining.optimizer = keras.optimizers.RMSprop(lr, momentum: 0.2f);
                PredictingTraining.optimizer = keras.optimizers.RMSprop(lr);
            }

            else if (optimizer == "adam")
            {
                PredictingTraining.optimizer = keras.optimizers.Adam(lr);
            }

            else
            {
                throw new Exception(message: "Invalid optimizer");
            }


            // Batch technique and size
            if (batch_mode == "random")
            {
                batch = tool.batch_random;
            }

            else if (batch_mode == "continuos")
            {
                batch = tool.batch_continuos;
            }

            else
            {
                throw new Exception(message: "Invalid batch mode!");
            }


            // Metrics of validation           
            if (metrics != null)
            {

                PredictingTraining.metrics = new List<(string name, int? param, Metric_Delegate method)>();

                foreach (var m in metrics)
                {
                    if (m.name == "rmse")
                    {
                        PredictingTraining.metrics.Add((m.name, m.param, prognostics.rmse_sequence));
                    }

                    else if (m.name == "rwmse")
                    {
                        PredictingTraining.metrics.Add((m.name, m.param, prognostics.rwmse_sequence));
                    }
                }
            }

            final_metrics_value = new List<(string metric, float value)>();


            // Eager executions
            tf.enable_eager_execution();


            // Train model
            Run();


            return (prediction_model, final_metrics_value);

        }

        private static void Run()
        {

            if (!(File.Exists(save_directory + "final_weights.hdf5")))
            {

                Directory.CreateDirectory(save_directory + @"training_weights\");

                var batch_indexes = batch(X, Y, batch_size);

                // Saving initial weights
                Console.WriteLine("Saving initial weigths");

                if (!(File.Exists(save_directory + "initial_weights.hdf5")))
                {
                    prediction_model.model.save_weights(save_directory + "initial_weights.hdf5");
                }

                else
                {
                    Console.WriteLine("Loading weigths...");
                    prediction_model.model.load_weights(save_directory + "initial_weights.hdf5");
                }

                // Epoch metrics
                var epoch_headers = new List<string> { "epoch", "batch", "time", "loss" };

                foreach (var m in metrics)
                {
                    epoch_headers.Add(m.name);
                }

                pd.to_csv(save_directory + "training_metrics.csv", headers: epoch_headers, append: false);

                // Training Time
                var time = new Stopwatch();
                time.Start();

                Console.WriteLine("Training...");


                for (int e = 0; e < epochs; e++)
                {

                    var epoch_best_loss = float.MaxValue;
                    var epoch_best_weigths = prediction_model.model.weights;
                    var epoch_metrics = new List<float[]>();


                    for (int i = 0; i < batch_indexes.Count; i++)
                    {

                        var batch_x = np.array<float>();
                        var batch_y = np.array<float>();


                        if (X.Count == 1)
                        {
                            var ind = new Slice(batch_indexes[i][0], batch_indexes[i][batch_indexes[i].Length - 1]);

                            batch_x = X[0].slice(ind).numpy();
                            batch_y = Y[0].slice(ind).numpy().reshape((-1, Y[0].shape[1]));
                        }

                        else if (X.Count > 1)
                        {
                            batch_x = X[batch_indexes[i][0]];
                            batch_y = Y[batch_indexes[i][0]].reshape((-1, Y[0].shape[1]));

                        }


                        run_optimization(optimizer, batch_x, batch_y);


                        // Loss
                        var pred = prediction_model.model.Apply(batch_x.reshape(prediction_model.model.Layers[0].output_shape));
                        var loss_value = np.mean(loss(pred, batch_y.reshape((prediction_model.model.output_shape)), RULwarning).numpy());


                        // Validation
                        var validation_metrics = new List<float>();

                        if (validation_x_sequence != null && validation_y_sequence != null && metrics != null)
                        {
                            var preds = new List<NDArray>();

                            foreach (var a in validation_x_sequence)
                            {
                                preds.Add(prediction_model.model.Apply(a.reshape(prediction_model.model.Layers[0].output_shape))[0].numpy());

                            }

                            foreach (var m in metrics)
                            {
                                validation_metrics.Add(m.method(preds, validation_y_sequence, m.param));
                            }

                        }


                        if (validation_metrics[0] < epoch_best_loss)
                        {
                            //Console.WriteLine("best");
                            epoch_best_loss = validation_metrics[0];
                            //epoch_best_weigths = prediction_model.model.weights;
                            prediction_model.model.save_weights(save_directory + @"training_weights\" + "temp_weights.hdf5");
                        }


                        // Store metrics
                        var temp_metrics = new List<float> { e + 1, i + 1, (float)time.Elapsed.TotalSeconds, loss_value };
                        temp_metrics.AddRange(validation_metrics);

                        epoch_metrics.Add(temp_metrics.ToArray());

                        // Print metrics
                        print(e, i, (float)loss_value, validation_metrics);


                        // Get final metrics values
                        if (e == epochs - 1 && i == batch_indexes.Count - 1)
                        {
                            foreach (var m in validation_metrics.Zip(metrics))
                            {
                                final_metrics_value.Add((m.Second.name, m.First));
                            }
                        }

                    }


                    // Save training metrics

                    pd.to_csv(save_directory + "training_metrics.csv", columns: np.array(epoch_metrics.ToArray().ToMatrix()), type: TypeCode.Empty);

                    // Save training weights

                    prediction_model.model.save_weights(save_directory + @"training_weights\" + "current_weights.hdf5");
                    //var current_weighs = prediction_model.model.weights;
                    prediction_model.model.load_weights(save_directory + @"training_weights\" + "temp_weights.hdf5");
                    //prediction_model.model.weights = epoch_best_weigths;
                    prediction_model.model.save_weights(save_directory + @"training_weights\" + "training_weights_" + (e + 1) + ".hdf5");

                    foreach (var w in prediction_model.model.trainable_variables)
                    {
                        pd.to_csv(save_directory + @"training_weights\" + "training_weights_" + (e + 1) + ".csv", columns: w.numpy());
                    }


                    prediction_model.model.load_weights(save_directory + @"training_weights\" + "current_weights.hdf5");
                    //prediction_model.model.weights = current_weighs;

                }

                // Save final weights
                prediction_model.model.save_weights(save_directory + "final_weights.hdf5");


            }

            else
            {
                prediction_model.model.load_weights(save_directory + "final_weights.hdf5");
            }


            Console.WriteLine("Train fisished");

        }

        private static void run_optimization(OptimizerV2 optimizer, Tensor x, Tensor y)
        {

            using var g = tf.GradientTape();

            var pred = prediction_model.model.Apply(tf.reshape(x, (prediction_model.model.Layers[0].output_shape)));

            var loss_value = loss(pred, y, RULwarning);

            var trainable_variables = prediction_model.model.trainable_variables;

            // Compute gradients.
            var gradients = g.gradient(loss_value, trainable_variables);

            // Update W and b following gradients.
            optimizer.apply_gradients(zip(gradients, trainable_variables.Select(x => x as ResourceVariable)));

        }

        private static void print(int epoch, int batch, float loss, List<float> metrics)
        {

            var coordenate = (Console.CursorLeft, Console.CursorTop);

            Console.SetCursorPosition(coordenate.CursorLeft + (15 * 0), coordenate.CursorTop);
            Console.Write("epoch: {0},", (epoch + 1).ToString("0.00", new CultureInfo("en-US")));
            Console.SetCursorPosition(coordenate.CursorLeft + (15 * 1), coordenate.CursorTop);
            Console.Write("batch: {0},", (batch + 1).ToString("0.00", new CultureInfo("en-US")));
            Console.SetCursorPosition(coordenate.CursorLeft + (15 * 2), coordenate.CursorTop);
            Console.Write("loss: {0},", ((float)loss).ToString("0.00"));

            for (int j = 0; j < metrics.Count; j++)
            {
                Console.SetCursorPosition(coordenate.CursorLeft + (16 * (3 + j)), coordenate.CursorTop);
                Console.Write("{0}: {1},", PredictingTraining.metrics[j].Item1, (metrics[j]).ToString("0.00"));
            }

            Console.WriteLine();
        }






        /*
         *  Delegates
         */


        delegate Tensor Loss_Delegate(Tensor pred, Tensor true_val, int? param);
        delegate List<int[]> Batch_Delegate(List<NDArray> x, List<NDArray> y, int batch_size);
        delegate float Metric_Delegate(List<NDArray> pred, List<NDArray> True, int? param);


    }
}
