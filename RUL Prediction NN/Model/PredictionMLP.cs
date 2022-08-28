using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.KerasApi;
using static Tensorflow.Binding;


namespace RUL_Prediction_NN.Model
{
    public class PredictionMLP : PredictionNN
    {


        public override Sequential model { get; set; }

        public PredictionMLP() : base()
        {

        }

        public void AddInputLayer(Shape input)
        {
            model.add(keras.layers.Input(input));
        }

        public void AddDenseLayer(int n_units, string activation, bool bn = false)
        {

            if (activation == "linear")
            {
                model.add(keras.layers.Dense(n_units, activation));
            }

            else if (activation != "leaky_relu")
            {
                if (bn)
                {
                    model.add(keras.layers.Dense(n_units, activation));
                    model.add(keras.layers.BatchNormalization());

                }

                else
                {
                    model.add(keras.layers.Dense(n_units, activation));
                }

            }

            else
            {
                if (bn)
                {
                    model.add(keras.layers.Dense(n_units));
                    model.add(keras.layers.LeakyReLU());
                    model.add(keras.layers.BatchNormalization());
                }

                else
                {
                    model.add(keras.layers.Dense(n_units));
                    model.add(keras.layers.LeakyReLU());
                }

            }
        }

        public override void Verify()
        {
            // Input Layer
            if (model.Layers[0].GetType().Name != "InputLayer")
            {
                throw new Exception(message: "First layer must be a input layer layer");
            }

            // Dims Input Layer
            if (model.Layers[0].output_shape.ndim != 2)
            {
                throw new Exception(message: "");
            }

            // Final Layer
            //if (model.Layers[model.Layers.Count - 1].GetType().Name != "Dense")
            //{
            //    throw new Exception(message: "Final layer must be a dense layer");
            //}

        }

        public override NDArray Predict(NDArray data, Func<NDArray, NDArray> gain = null)
        {

            var prediction = (model.Apply(data.reshape(model.Layers[0].output_shape))[0].numpy());

            if (gain == null)
            {
                return prediction;
            }

            else
            {
                return gain(prediction);
            }

        }


    }
}
