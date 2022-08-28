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
    public abstract class PredictionNN : IPredictionModel
    {

        public abstract Sequential model { get; set; }

        public PredictionNN()
        {
            model = keras.Sequential();
        }

        public abstract NDArray Predict(NDArray data, Func<NDArray, NDArray> gain = null);

        public void Summary()
        {
            model.summary();
        }

        public abstract void Verify();

        public void CLear()
        {
            model = keras.Sequential();
        }

    }
}
