using Accord.Math;
using Accord.Statistics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RUL_Prediction_NN.Misc
{
    public static class preprocessing
    {


        public static float[,] MinMaxScaler(float[,] data, (int, int) range, List<int> cols_id = null)
        {

            // norm_value = value - min_value / max_value - min_value

            var _data = new float[data.Columns(), data.Rows()];

            if (!(cols_id == null))
            {
                //Get columns to normalize defined by cols_id
                _data = data.GetColumns(cols_id.ToArray());
            }


            float valor_max = 0.0f;
            float valor_min = 0.0f;

            // Trasnpose matrix to normalize the columns
            var _data_transpose = _data.Transpose();

            for (int i = 0; i < _data_transpose.Rows(); i++)
            {
                valor_max = _data_transpose.GetRow(i).Max();
                valor_min = _data_transpose.GetRow(i).Min();

                for (int j = 0; j < _data_transpose.Columns(); j++)
                {
                    if (valor_max - valor_min == 0)
                    {
                        _data_transpose[i, j] = 0;
                    }
                    else
                    {
                        _data_transpose[i, j] = (_data_transpose[i, j] - valor_min) / (valor_max - valor_min);
                    }

                }
            }

            // Range scaler
            for (int i = 0; i < _data_transpose.Rows(); i++)
            {
                for (int j = 0; j < _data_transpose.Columns(); j++)
                {
                    _data_transpose[i, j] = _data_transpose[i, j] * range.Item2;

                }
            }

            _data = _data_transpose.Transpose();


            if (!(cols_id == null))
            {
                // Add columns not normalized
                for (int i = 0; i < cols_id.Count; i++)
                {
                    data.SetColumn(cols_id[i], _data.GetColumn(i));
                }
            }


            return data;


        }

        public static float[,] StandardScaler(float[,] data, List<int> cols_id)
        {
            // norm_value = value - mean / standard_deviation

            //Get columns to normalize defined by cols_id
            var _data = data.GetColumns(cols_id.ToArray());

            // Trasnpose matrix to normalize the columns
            var _data_transpose = _data.Transpose();

            for (int i = 0; i < _data_transpose.Rows(); i++)
            {

                var media = _data_transpose.GetRow(i).Mean();
                var std = _data_transpose.GetRow(i).StandardDeviation();

                for (int j = 0; j < _data_transpose.Columns(); j++)
                {

                    if (std == 0)
                    {
                        _data_transpose[i, j] = 0;
                    }

                    else
                    {
                        _data_transpose[i, j] = ((_data_transpose[i, j] - (float)media) / ((float)std));
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

