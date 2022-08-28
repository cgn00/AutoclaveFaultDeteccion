using Accord.Math;
using Accord.Statistics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RUL_Prediction_NN.Transformation
{
    public class MinMax : ITransformation
    {

        // minmax normalization

        public double[] max;
        public double[] min;


        public void Fit(params object[] parameters)
        {

            var data = (float[,])parameters[0];

            var cols_id = new List<int>();
            cols_id.AddRange(Enumerable.Range(2, data.Columns() - 2));

            var _data = data.GetColumns(cols_id.ToArray());


            // Trasnpose matrix to normalize the columns
            var _data_transpose = _data.Transpose();

            min = new double[_data_transpose.Rows()];
            max = new double[_data_transpose.Rows()];

            for (int i = 0; i < _data_transpose.Rows(); i++)
            {

                min[i] = _data_transpose.GetRow(i).Min();
                max[i] = _data_transpose.GetRow(i).Max();

            }

        }

        public float[,] Transform(float[,] data)
        {

            // norm_value = value - min_value / max_value - min_value

            var cols_id = new List<int>();
            cols_id.AddRange(Enumerable.Range(2, data.Columns() - 2));

            var _data = data.GetColumns(cols_id.ToArray());

            // Trasnpose matrix to normalize the columns
            var _data_transpose = _data.Transpose();

            for (int i = 0; i < _data_transpose.Rows(); i++)
            {

                for (int j = 0; j < _data_transpose.Columns(); j++)
                {
                    if (max[i] - min[i] == 0)
                    {
                        _data_transpose[i, j] = 0;
                    }
                    else
                    {
                        _data_transpose[i, j] = (float)((_data_transpose[i, j] - min[i]) / (max[i] - min[i]));
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
