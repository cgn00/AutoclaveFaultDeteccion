using System.Collections.Generic;
using Tensorflow;
using static Tensorflow.Binding;
using Tensorflow.NumPy;

namespace RUL_Prediction_NN.Misc
{

    public static class prognostics
    {


        public static Tensor square_error(Tensor pred, Tensor True, int? param = null)
        {
            return (tf.square(pred - True));
        }

        public static Tensor weighted_square_error(Tensor pred, Tensor True, int? param = null)
        {
            return (exponential_weight_tf(True, (int)param) * tf.square(pred - True));
        }

        private static Tensor exponential_weight_tf(Tensor rul, int rul_warning)
        {
            var a = 1.0f;
            var b = -1.0f / (rul_warning + 1.0f);
            var c = 0.0f;
            var x = rul - rul_warning;
            var weights = a * tf.exp(b * x) + c;

            return weights;
        }






        // Exponential weigths
        static NDArray exponential_weight(NDArray rul, int rul_warning)
        {

            float a = 1.0f;
            float b = -1.0f / (rul_warning + 1.0f);
            float c = 0.0f;
            var x = rul - rul_warning;
            var weights = a * np.exp(b * x) + c;

            return weights;

        }






        // Validations functions
        public static float rwmse_sequence(List<NDArray> pred_values, List<NDArray> true_values, int? param)
        {

            //root weighted mean square error

            var n_output = pred_values[0].shape[1];

            var sequences_count = true_values.Count;
            var wsum = 0.0f;

            for (int i = 0; i < sequences_count; i++)
            {

                var weigths_seq = exponential_weight(true_values[i], (int)param).reshape((-1, n_output));
                var error = pred_values[i].reshape((-1, n_output)) - true_values[i].reshape((-1, n_output));
                var prod = np.multiply(np.multiply(error, error), weigths_seq);
                wsum = wsum + np.sqrt(np.sum(prod) / np.sum(weigths_seq));

            }

            return (float)(wsum / sequences_count);

        }

        public static float rmse_sequence(List<NDArray> pred_values, List<NDArray> true_values, int? param)
        {
            //root mean square error

            var n_output = pred_values[0].shape[1];

            var sequences_count = true_values.Count;

            var wsum = 0.0f;

            for (int i = 0; i < sequences_count; i++)
            {
                var error = pred_values[i].reshape((-1, n_output)) - true_values[i].reshape((-1, n_output));
                var prod = np.multiply(error, error);
                wsum = wsum + np.sqrt(np.sum(prod) / prod.shape[0]);
            }

            return (float)(wsum / sequences_count);

        }








    }
}
