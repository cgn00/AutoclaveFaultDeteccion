using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RUL_Prediction_NN.data_model
{
    public class Alarm
    {

        public int EntityId { get; set; }

        public int DeviceId { get; set; }

        public DateTime Time { get; set; }

        public string Text { get; set; }

        public string Discriminator { get; set; }

    }
}
