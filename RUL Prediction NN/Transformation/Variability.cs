using Accord.Math;
using Accord.Statistics;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.NumPy;
using MathNet.Numerics.Statistics;
using RUL_Prediction_NN.Data;
using System.IO;

namespace RUL_Prediction_NN.Transformation
{
    public class Variability : ITransformation
    {

        // weigths of metrics
        private double wp = 1;
        private double wm = 1;
        private double wt = 0;

        // Thereshold for suitable variables
        private double thereshold = 1.0;


        // Intervals to compute monotonicity
        private int window_size = 10;


        // Length to interpolate in correlation
        private int len = 100;


        // Path
        private string directory = @"C:\Users\Ale\Desktop\4.2\Tesis\Analysis of Executions\Producción principal\";



        public void Fit(params object[] parameters)
        {

            /*
             *  Parameters:
             *  1- path
             *  2- wp
             *  3- wm
             *  4- wt
             *  5- theresold
             *  6- window_size (for monotonicity)
             */


            directory = (string)parameters[0];

            wp = (double)parameters[1];
            wm = (double)parameters[2];
            wt = (double)parameters[3];

            thereshold = (double)parameters[4];

            window_size = (int)parameters[5];

            len = (int)parameters[6];

        }

        public float[,] Transform(float[,] data)
        {

            // Divide samples for sequences

            var data_seq = new List<float[,]>();

            var index = data.ToJagged().DistinctCount()[0];

            for (int i = 1; i <= index; i++)
            {
                var seq = from x in data.ToJagged()
                          where x[0] == i
                          select (x);

                // Variables columns
                data_seq.Add(seq.ToArray().ToMatrix());


            }

            // Metrics of variability

            var p = new List<double>();
            var t = new List<double>();
            var m = new List<double>();


            if (File.Exists(directory + "prognosability.csv") && File.Exists(directory + "monotonicity.csv") && File.Exists(directory + "trendability.csv"))
            {
                var _p = pd.read_csv(directory + "prognosability.csv");
                var _m = pd.read_csv(directory + "monotonicity.csv");
                var _t = pd.read_csv(directory + "trendability.csv");

                for (int i = 0; i < _p.ToJagged()[0].Length; i++)
                {
                    p.Add(Convert.ToDouble(_p[0, i]));
                    t.Add(Convert.ToDouble(_t[0, i]));
                    m.Add(Convert.ToDouble(_m[0, i]));
                }
            }

            else
            {
                p = Prognosability(data_seq);
                m = Monotonicity(data_seq);
                t = Trendability(data_seq);

                //t = new List<double> { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

                // Save metrics

                pd.to_csv(directory + "prognosability.csv", columns: np.array(p.ToArray()), type: TypeCode.Double, append: false);
                pd.to_csv(directory + "monotonicity.csv", columns: np.array(m.ToArray()), type: TypeCode.Double, append: false);
                pd.to_csv(directory + "trendability.csv", columns: np.array(t.ToArray()), type: TypeCode.Double, append: false);

            }


            // Suitable variables
            var suitables = (p.ToArray().Multiply(wp).Add(t.ToArray().Multiply(wt))).Add(m.ToArray().Multiply(wm));

            var good_parameter = new List<int> { 0, 1 };

            for (int i = 0; i < suitables.Length; i++)
            {
                if (suitables[i] > thereshold)
                {
                    good_parameter.Add(i + 2);
                }
            }

            return data.ToJagged().GetColumns(good_parameter.ToArray()).ToMatrix();


        }




        List<double> Trendability(List<float[,]> data)
        {

            var trendability = new List<double>();

            for (int v = 2; v < data[0].Columns(); v++)
            {

                var distance = new List<double>();

                for (int j = 0; j < data.Count; j++)
                {
                    var temp_corr = new List<double>();

                    for (int k = 0; k < data.Count; k++)
                    {

                        // Select executions
                        var r1 = data[j].ToJagged(transpose: true)[v];
                        var r2 = data[k].ToJagged(transpose: true)[v];

                        // Convert to duoble
                        var r3 = Array.ConvertAll(r1, x => (double)x);
                        var r4 = Array.ConvertAll(r2, x => (double)x);

                        // CorrCoef
                        var temp = Correlation.Pearson(Resample(r3, len), Resample(r4, len));
                        temp_corr.Add(temp);


                    }

                    distance.Add(temp_corr.MinimumAbsolute());

                }

                trendability.Add(distance.ToArray().Min());

            }


            for (int u = 0; u < trendability.Count; u++)
            {
                if (double.IsNaN(trendability[u]) == true)
                {
                    trendability[u] = 0;
                }
            }


            return trendability;


        }

        List<double> Prognosability(List<float[,]> data)
        {

            var prognosability = new List<double>();

            for (int v = 2; v < data[0].Columns(); v++)
            {

                var failval = new List<float>();
                var startlval = new List<float>();

                for (int j = 0; j < data.Count; j++)
                {
                    // Final value
                    var t_f = (data[j][data[j].Rows() - 1, v]);
                    failval.Add(t_f);

                    // Start value
                    var t_s = (data[j][0, v]);
                    startlval.Add(t_s);
                }

                // Standard deviation
                var std = failval.ToArray().StandardDeviation();

                // Mean
                var mean = startlval.ToArray().Subtract(failval.ToArray()).Abs().Mean();

                prognosability.Add(Math.Exp(-std / mean));

            }


            for (int u = 0; u < prognosability.Count; u++)
            {
                if (double.IsNaN(prognosability[u]) == true)
                {
                    prognosability[u] = 0;
                }
            }

            return prognosability;

        }

        List<double> Monotonicity(List<float[,]> data)
        {

            var monotonicity = new List<double>();

            for (int v = 2; v < data[0].Columns(); v++)
            {
                var sum = 0.0;

                for (int j = 0; j < data.Count; j++)
                {

                    var PD = 0;
                    var ND = 0;

                    for (int k = 0; k < data[j].Rows() - window_size; k++)
                    {

                        var d1 = data[j].ToJagged(true)[v][k + window_size];
                        var d2 = data[j].ToJagged(true)[v][k];

                        if ((d1 - d2) > 0)
                        {
                            PD++;
                        }
                        else if ((d1 - d2) < 0)
                        {
                            ND++;
                        }
                    }

                    sum += Math.Abs((PD - ND) / (double)(data[j].Rows() - window_size));


                }

                monotonicity.Add(sum / data.Count);

            }

            for (int u = 0; u < monotonicity.Count; u++)
            {
                if (double.IsNaN(monotonicity[u]) == true)
                {
                    monotonicity[u] = 0;
                }
            }

            return monotonicity;
        }






        // Linear Interpolation
        static double[] Resample(double[] source, int n)
        {
            int m = source.Length;
            var destination = new double[n];
            destination[0] = source[0];
            destination[n - 1] = source[m - 1];

            for (int i = 1; i < n - 1; i++)
            {
                var jd = ((double)i * (double)(m - 1) / (double)(n - 1));
                var j = (int)jd;
                destination[i] = (double)Math.Round((source[j] + (source[j + 1] - source[j]) * (jd - (double)j)), 2);
            }

            return destination;

        }

    }
}
