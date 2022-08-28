using System;
using System.Collections.Generic;
using Tensorflow.Keras.Engine;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using Tensorflow.NumPy;
using Tensorflow;
using System.IO;

namespace RUL_Prediction_NN
{
    public class autoencoder
    {

        Sequential _model_ae;

        NDArray _X, _Y;

        OptimizerV2 _optimizer;

        ILossFunc _loss;

        List<string> _metrics = new List<string> { "accuracy" };

        float[] _learning_reate = { 0.01f };
        float _validation_split = 0.1f;

        int _n_input, _n_hidden, _n_output;
        Shape _input_shape;

        int _epochs = 50;
        int _batch_size = 50;

        string _model_name;


        string _save_directory = @".\results\ae\";




        public autoencoder(string model, NDArray X)
        {
            _model_name = model;

            _X = X;
            _Y = X;

            _n_input = (int)_X.shape[1];
            _n_output = (int)_X.shape[1];
            _input_shape = (_n_input);
        }

        private void ae()
        {
            _model_ae = keras.Sequential();
            _model_ae.add(keras.Input(_input_shape));
            _model_ae.add(keras.layers.Dense(_n_hidden, activation: keras.activations.Relu));
            _model_ae.add(keras.layers.Dense(_n_output, activation: keras.activations.Sigmoid));

        }

        public void Run(int[] n_variables)
        {

            foreach (var lr in _learning_reate)
            {

                _optimizer = keras.optimizers.Adam(lr);
                _loss = keras.losses.MeanSquaredError();

                foreach (var i in n_variables)
                {

                    _n_hidden = i;

                    ae();

                    _model_ae.summary();

                    if (!Directory.Exists(_save_directory + _model_name + @"\" + "variables_" + i + @"\"))
                    {
                        Directory.CreateDirectory(_save_directory + _model_name + @"\" + "variables_" + i + @"\");
                    }

                    _model_ae.save_weights(_save_directory + _model_name + @"\" + "variables_" + i + @"\" + "initial_weights.hdf5");

                    _model_ae.compile(optimizer: _optimizer, loss: _loss, metrics: _metrics.ToArray());

                    _model_ae.fit(_X, _Y, epochs: _epochs, batch_size: _batch_size, validation_split: _validation_split);

                    _model_ae.save_weights(_save_directory + _model_name + @"\" + "variables_" + i + @"\" + "final_weights.hdf5");


                    Console.WriteLine("Variables: {0}, learning rate: {1}", i, lr);
                    Console.WriteLine("Train finished!");
                    Console.ReadLine();
                    Console.Clear();

                }


            }


        }



    }

}
