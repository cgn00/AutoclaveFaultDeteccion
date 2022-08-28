using Accord.Math;
using RUL_Prediction_NN.Data;
using RUL_Prediction_NN.Misc;
using RUL_Prediction_NN.Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace RUL_Prediction_NN.Train
{
    public class CrossValidation
    {


        private int[][] partitions;


        string save_directory;
        string model_name;

        (string, int?) loss_method;

        string optimizer;
        float lr;

        int epochs;


        string batch_mode;
        int batch_size;


        List<(string name, int? param)> metrics;


        string data_presentation;




        public CrossValidation(int k, List<NDArray> data, params object[] parameters)
        {

            model_name = (string)parameters[0];
            epochs = (int)parameters[1];
            loss_method = ((string, int?))parameters[2];
            optimizer = (string)parameters[3];
            lr = (float)parameters[4];
            batch_mode = (string)parameters[5];
            batch_size = (int)parameters[6];
            metrics = (List<(string name, int? param)>)parameters[7];
            data_presentation = (string)parameters[8];

            save_directory = Directory.GetCurrentDirectory() + @"\results\" + model_name;

            var data_len = data.Count;

            if (!(File.Exists(save_directory + @"\partitions.csv")))
            {

                if (!Directory.Exists(save_directory))
                {
                    Directory.CreateDirectory(save_directory);
                }

                var data_indexes = Enumerable.Range(0, data_len).ToList();

                data_indexes.Shuffle();

                var temp_partitions = data_indexes.ToArray().Reshape(k, data_len / k).ToJagged().ToList();

                var partition_dataframe = np.array(temp_partitions.ToArray().ToMatrix());
                pd.to_csv(save_directory + @"\partitions.csv", columns: partition_dataframe, type: TypeCode.Int32);

                partitions = temp_partitions.ToArray();
            }

            else
            {
                var _partitions = pd.read_csv(save_directory + @"\partitions.csv").ToJagged().ToList();
                var temp_partitions = new List<int[]>();

                for (int p = 0; p < _partitions.Count; p++)
                {
                    temp_partitions.Add(new int[_partitions[p].Length]);

                    for (int i = 0; i < _partitions[p].Length; i++)
                    {
                        temp_partitions[p][i] = Convert.ToInt32(_partitions[p][i]);
                    }
                }

                partitions = temp_partitions.ToArray();
            }


        }

        public void Execute(IPredictionModel model, List<NDArray> X, List<NDArray> Y, List<NDArray> sequences_id, List<NDArray> times_id)
        {

            var validation_directory = save_directory + @"\cross_validation\";

            if (!(Directory.Exists(validation_directory)))
            {
                Directory.CreateDirectory(validation_directory);
            }

            // Model output testing
            var headers = new List<string> { "partition", "sequence", "time", "true", "predicted" };
            pd.to_csv(validation_directory + "model_output_testing.csv", headers: headers, append: false);

            // Initial weigths
            model.model.save_weights(validation_directory + "initial_weigths.hdf5");

            // Cross-validation
            for (int p = 0; p < partitions.Rows(); p++)
            {

                var partition_directory = validation_directory + @"partition_" + p + @"\";

                Console.WriteLine("Training partition {0}", p);

                // Select data for traininntg 
                var _train_part = new List<int[]>();

                for (int j = 0; j < partitions.Rows(); j++)
                {
                    // Exclude validation data 
                    if (!(j == p))
                    {
                        _train_part.Add(partitions[j]);
                    }
                }

                var train_part = _train_part.ToArray().Concatenate();

                var x_part = new List<NDArray>();
                var y_part = new List<NDArray>();

                foreach (var ipart in train_part)
                {
                    x_part.Add(X[ipart]);
                    y_part.Add(Y[ipart]);
                }

                var x_train = new List<NDArray>();
                var y_train = new List<NDArray>();

                if (data_presentation == "sample")
                {
                    x_train.Add(np.concatenate(x_part.ToArray()));
                    y_train.Add(np.concatenate(y_part.ToArray()));
                }

                else if (data_presentation == "sequence")
                {
                    x_train = x_part;
                    y_train = y_part;
                }

                else
                {
                    throw new Exception(message: "");
                }

                // Training model

                model.model.load_weights(validation_directory + "initial_weigths.hdf5");

                var (pred_model, _) = PredictingTraining.Train(model: model, data: x_train, label: y_train,
                                                      partition_directory,
                                                      epochs,
                                                      loss_method,
                                                      optimizer,
                                                      lr,
                                                      batch_mode,
                                                      batch_size,
                                                      metrics,
                                                      X,
                                                      Y);


                // Select data for testing
                var test_part = partitions[p];

                var x_part_test = new List<NDArray>();
                var y_part_test = new List<NDArray>();
                var sequenceces_id_part_test = new List<NDArray>();
                var time_id_part_test = new List<NDArray>();

                foreach (var j in test_part)
                {
                    x_part_test.Add(X[j]);
                    y_part_test.Add(Y[j]);
                    sequenceces_id_part_test.Add(sequences_id[j]);
                    time_id_part_test.Add(times_id[j]);

                }

                //var partition_id = np.repeat(p + 1, np.concatenate(time_id_part_test.ToArray()).shape[0]).reshape((-1, 1));
                var partition_id = (np.ones((np.concatenate(time_id_part_test.ToArray()).shape[0], 1), tf.float32) * (p + 1)).reshape((-1, 1));

                // Testing model and saving output

                var temp_preds = new List<NDArray>();

                foreach (var a in x_part_test)
                {
                    temp_preds.Add(pred_model.model.Apply(a.reshape(pred_model.model.Layers[0].output_shape))[0].numpy());
                }

                var preds = np.concatenate(temp_preds.ToArray());
                var trues = np.concatenate(y_part_test.ToArray());
                var sequences = np.concatenate(sequenceces_id_part_test.ToArray());
                var time = np.concatenate(time_id_part_test.ToArray());


                // Saving results
                Console.WriteLine("Saving results...");

                tool.save_model_output_prediction(save_directory + @"\cross_validation\model_output_testing.csv", trues, preds, sequences, time, partition_id);

                Console.WriteLine("Saving Predictions fisished");


            }


        }


    }
}
