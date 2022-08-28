using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.NumPy;

namespace RUL_Prediction_NN.Data
{
    public class DataRead
    {

        public static (float[,], float[,]) LoadData()
        {

            // String connections
            string string_label_connection = @".\Datasource\label.csv";
            string string_data_connection = @".\Datasource\data.csv";

            // Loading datas
            var data = pd.read_csv(string_data_connection);

            //Loading labels        
            var label = pd.read_csv(string_label_connection);

            return (data, label);
        }

        public static (List<NDArray>, List<NDArray>, List<NDArray>, List<NDArray>) GetSequences(float[,] data, float[,] label, IEnumerable<int> variables = null)
        {

            var data_seq = new List<NDArray>();
            var label_seq = new List<NDArray>(); // quien es label???
            var seq_id = new List<NDArray>();
            var tim_id = new List<NDArray>();

            // por q se convierte data a matriz irregular????
            var pp = data.ToJagged(); // por q no se usa pp y se siguen haciendo llamadas a la función data.ToJagged()????????

            var index = data.ToJagged().Distinct()[0]; // array con los SequenceID existentes(no repetidos)

            foreach (var i in index)
            {
                // seq: es un IEnumerable q contiene todas las secuencias con el mismo ID 
                var seq = from x in data.ToJagged() // se va buscando en cada fila el ID y se guarda esta si ID == i
                          where x[0] == i
                          select (x);

                // Sequence ID
                seq_id.Add(seq.ToArray().GetColumn(0));

                // Time ID
                tim_id.Add(seq.ToArray().GetColumn(1)); // quien es time ID???

                // Variables columns
                if (variables != null)
                {
                    data_seq.Add(seq.ToArray().GetColumns(variables.ToArray()).ToMatrix());
                }

                else
                {
                    data_seq.Add(seq.ToArray().GetColumns((Enumerable.Range(2, data.Columns() - 2)).ToArray()).ToMatrix());
                }

            }

            index = label.ToJagged().Distinct()[0];

            foreach (var i in index)
            {
                var seq = from x in label.ToJagged()
                          where x[0] == i
                          select x;

                // Labels
                label_seq.Add(np.array(seq.ToArray().GetColumn(1)).reshape((-1, 1)));
            }

            return (data_seq, label_seq, seq_id, tim_id);

        }

        public static (List<NDArray>, List<NDArray>, List<NDArray>, List<NDArray>) SlideWindow(List<NDArray> data, List<NDArray> label, List<NDArray> seqs, List<NDArray> times, int window_size)
        {

            /*
             *  Reshape to intervals for slice window in data
             */

            data = reshape_sequences_to_intervals(data, window_size);

            for (int i = 0; i < label.Count; i++)
            {
                label[i] = label[i][new Slice(window_size - 1, (int)label[i].shape[0])];
                seqs[i] = seqs[i][new Slice(window_size - 1, (int)seqs[i].shape[0])];
                times[i] = times[i][new Slice(window_size - 1, (int)times[i].shape[0])];
            }

            return (data, label, seqs, times);

        }

        private static List<NDArray> reshape_sequences_to_intervals(List<NDArray> data_sequences, int interval_size)
        {

            var reshaped_sequences = new List<NDArray>();

            foreach (var seq in data_sequences)
            {
                var seq_size = seq.shape[0];
                var reshaped_seq = new List<NDArray>();

                for (int i = interval_size; i < seq_size + 1; i++)
                {
                    var interval = seq[new Slice(i - interval_size, i), ":"];
                    reshaped_seq.Add(interval);
                }

                var temp = np.concatenate(reshaped_seq.ToArray());
                reshaped_sequences.Add(temp.reshape((-1, interval_size, seq.shape[1])));

            }


            return reshaped_sequences;

        }


    }


}
