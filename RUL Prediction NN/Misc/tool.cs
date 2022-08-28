using Accord.Math;
using RUL_Prediction_NN.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.NumPy;

namespace RUL_Prediction_NN.Misc
{
    public static class tool
    {

        /*
         * Training and prediction functions
         */


        public static List<int[]> batch_random(List<NDArray> x, List<NDArray> y, int batch_size)
        {
            var data_len = 0;

            // Data Presentation Samples
            if (x.Count == 1)
            {
                data_len = (int)x[0].shape[0];
            }

            // Data Presentation Sequences
            else
            {
                data_len = x.Count;
            }

            var batch_count = Decimal.Ceiling(data_len / batch_size);         //Divide y redondea por encima
            var data_indexes = new List<int>(Enumerable.Range(0, data_len));

            // Randomize and resize
            data_indexes.Shuffle();

            var final_indexes = data_indexes.GetRange((int)(batch_count * batch_size), (data_indexes.Count - ((int)batch_count * batch_size)));

            data_indexes = (data_indexes.GetRange(0, (int)(batch_count * batch_size)));

            var batches = data_indexes.ToArray().Reshape((int)batch_count, batch_size).ToJagged().ToList();

            if (final_indexes.Count != 0)
            {
                batches.Add(final_indexes.ToArray());
            }

            return batches;
        }

        public static List<int[]> batch_continuos(List<NDArray> x, List<NDArray> y, int batch_size)
        {

            /*
             *  Divide data into continuos batches
             */

            var data_len = 0;

            // Data Presentation Samples
            if (x.Count == 1)
            {
                data_len = (int)x[0].shape[0];
            }

            // Data Presentation Sequences
            else
            {
                data_len = x.Count;
            }

            var batch_count = Decimal.Ceiling(data_len / batch_size);         //Divide y redondea por encima
            var data_indexes = new List<int>(Enumerable.Range(0, data_len));

            // Resize

            var final_indexes = data_indexes.GetRange((int)(batch_count * batch_size), (data_indexes.Count - ((int)batch_count * batch_size)));

            data_indexes = (data_indexes.GetRange(0, (int)(batch_count * batch_size)));

            var batches = data_indexes.ToArray().Reshape((int)batch_count, batch_size).ToJagged().ToList();

            if (final_indexes.Count != 0)
            {
                batches.Add(final_indexes.ToArray());
            }

            return batches;

        }

        public static void save_model_output_prediction(string save_directory, NDArray true_values, NDArray predicted_values, NDArray seq_id = null, NDArray time_id = null, NDArray partition = null)
        {

            /*
             *  Saving predictions and truevalues in csv file
             */


            Console.WriteLine("Saving model output prediction... ");


            // To float
            var _true_values = true_values.ToArray<float>();
            var _predicted = predicted_values.ToArray<float>();

            var _seqs = new List<float>();
            var _times = new List<float>();
            var _partition = new List<float>();


            if (seq_id.ToMultiDimArray<float>().Length > 1)
            {
                _seqs = seq_id.ToArray<float>().ToList();
            }

            if (time_id.ToMultiDimArray<float>().Length > 1)
            {
                _times = time_id.ToArray<float>().ToList();
            }

            if (partition.ToMultiDimArray<float>().Length > 1)
            {
                _partition = partition.ToArray<float>().ToList();
            }

            // To save

            if (!(seq_id.ToMultiDimArray<float>().Length > 1) && !(time_id.ToMultiDimArray<float>().Length > 1) && !(partition.ToMultiDimArray<float>().Length > 1))
            {
                var temp = new float[2][] { _true_values, _predicted };
                var tosave = np.array(temp.ToMatrix(transpose: true));

                // Saving
                pd.to_csv(save_directory, columns: tosave);
            }

            else if (!(partition.ToMultiDimArray<float>().Length > 1))
            {
                var temp = new float[4][] { _seqs.ToArray(), _times.ToArray(), _true_values, _predicted };
                var tosave = np.array(temp.ToMatrix(transpose: true));

                // Saving
                pd.to_csv(save_directory, columns: tosave);
            }

            else
            {
                var temp = new float[5][] { _partition.ToArray(), _seqs.ToArray(), _times.ToArray(), _true_values, _predicted };
                var tosave = np.array(temp.ToMatrix(transpose: true));

                // Saving
                pd.to_csv(save_directory, columns: tosave);
            }

            Console.WriteLine("Prediction saved");

        }









    }
}
