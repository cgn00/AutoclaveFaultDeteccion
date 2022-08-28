using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RUL_Prediction_NN.data_model
{
    public class Sample
    {

        public int TimeId { get; set; }
        public DateTime Time { get; set; }

        public double Value { get; set; }

        public int VariableId { get; set; }

        public string Name { get; set; }

    }
}
