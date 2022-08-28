using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RUL_Prediction_NN.Transformation
{
    public interface ITransformation
    {

        void Fit(params object[] parammeters);

        float[,] Transform(float[,] data);

    }
}
