using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.NumPy;
using Tensorflow.Keras.Engine;
using static Tensorflow.KerasApi;

namespace RUL_Prediction_NN.Model
{
    public class PredictionCNN : PredictionNN
    {


        public override Sequential model { get; set; }

        public PredictionCNN() : base()
        {

        }

        public void AddInputLayer(Shape input)
        {
            model.add(keras.layers.Input(input));
        }

        public void AddConvLayer(int n_filters, Shape kernel_size, string activation)
        {
            if (activation != "leaky_relu")
            {
                model.add(keras.layers.Conv2D(n_filters, kernel_size, activation: activation, padding: "same"));
            }

            else
            {
                model.add(keras.layers.Conv2D(n_filters, kernel_size, activation: "linear", padding: "same"));
                model.add(keras.layers.LeakyReLU());
            }
        }

        public void AddFlattenLayer()
        {
            model.add(keras.layers.Flatten());
        }

        public void AddDropoutLayer(float dropout)
        {
            model.add(keras.layers.Dropout(dropout));
        }

        public void AddDenseLayer(int n_units, string activation)
        {
            if (activation != "leaky_relu")
            {
                model.add(keras.layers.Dense(n_units, activation));
            }

            else
            {
                model.add(keras.layers.Dense(n_units));
                model.add(keras.layers.LeakyReLU());
            }

        }

        public void AddPoolingLayer(Shape pooling)
        {
            model.add(keras.layers.MaxPooling2D(pool_size: pooling, padding: "same"));
        }

        public override NDArray Predict(NDArray data, Func<NDArray, NDArray> gain = null)
        {

            // arreglar

            //var prediction = (model.Apply(data.reshape(model.Layers[0].output_shape))[0].numpy());
            var prediction = (model.Apply(data)[0].numpy());

            if (gain == null)
            {
                return prediction;
            }

            else
            {
                return gain(prediction);
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
            if (model.Layers[0].output_shape.ndim != 4)
            {
                throw new Exception(message: "");
            }

            // Final Layer
            //if (model.Layers[model.Layers.Count - 1].GetType().Name != "Dense")
            //{
            //    throw new Exception(message: "Final layer must be a dense layer");
            //}

            // Flatten Layer
            for (int i = 0; i < model.Layers.Count; i++)
            {
                if (model.Layers[i].GetType().Name == "Flatten")
                {
                    if ((model.Layers[i - 1].GetType().Name != "Conv2D" && model.Layers[i - 1].GetType().Name != "LeakyReLu"))
                    {
                        throw new Exception(message: "");
                    }
                }
            }

        }


    }
}
