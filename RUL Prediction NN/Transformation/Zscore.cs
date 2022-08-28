using Accord.Math;
using Accord.Statistics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RUL_Prediction_NN.Transformation
{
    public class Zscore : ITransformation
    {

        // z-score normalization

        public double[] std;
        public double[] mean;


        public void Fit(params object[] parameters)
        {

            var data = (float[,])parameters[0];

            var cols_id = new List<int>();
            cols_id.AddRange(Enumerable.Range(2, data.Columns() - 2));

            var _data = data.GetColumns(cols_id.ToArray());

            // Trasnpose matrix to normalize the columns
            var _data_transpose = _data.Transpose();

            mean = new double[_data_transpose.Rows()];
            std = new double[_data_transpose.Rows()];

            for (int i = 0; i < _data_transpose.Rows(); i++)
            {

                mean[i] = _data_transpose.GetRow(i).Mean();
                std[i] = _data_transpose.GetRow(i).StandardDeviation();

            }

        }

        public float[,] Transform(float[,] data)
        {
            // norm_value = value - mean / standard_deviation

            var cols_id = new List<int>();
            cols_id.AddRange(Enumerable.Range(2, data.Columns() - 2));

            var _data = data.GetColumns(cols_id.ToArray());

            // Trasnpose matrix to normalize the columns
            var _data_transpose = _data.Transpose();

            for (int i = 0; i < _data_transpose.Rows(); i++)
            {

                for (int j = 0; j < _data_transpose.Columns(); j++)
                {

                    if (std[i] < 0.0000001)
                    {
                        _data_transpose[i, j] = 0;
                    }

                    else
                    {
                        _data_transpose[i, j] = ((_data_transpose[i, j] - (float)mean[i]) / ((float)std[i]));
                    }


                }
            }

            _data = _data_transpose.Transpose();


            // Add columns not normalized
            for (int i = 0; i < cols_id.Count; i++)
            {
                data.SetColumn(cols_id[i], _data.GetColumn(i));
            }

            return data;
        }


    }
}
