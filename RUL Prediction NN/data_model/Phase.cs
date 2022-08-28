using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RUL_Prediction_NN.data_model
{
    public class Phase
    {

        public int PhaseId { get; set; }

        public int EntityId { get; set; }

        public int ExecutionId { get; set; }

        public int OperatorId { get; set; }

        public DateTime Time { get; set; }

        public string Text { get; set; }

        public (DateTime, DateTime) Duration { get; set; }

        public double InitialConductivity { get; set; }


    }
}
