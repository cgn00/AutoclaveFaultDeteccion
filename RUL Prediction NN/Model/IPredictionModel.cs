using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.KerasApi;

namespace RUL_Prediction_NN.Model
{
    public interface IPredictionModel
    {

        public Sequential model { get; set; }

        public NDArray Predict(NDArray data, Func<NDArray, NDArray> gain = null);


    }
}
