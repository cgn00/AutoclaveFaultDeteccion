using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RUL_Prediction_NN.data_model
{
    public class Execution
    {

        public int DefinitionId { get; set; }

        public int ExecutionId { get; set; }

        public DateTime StartDate { get; set; }

        public DateTime EndDate { get; set; }

        public int? StartingOperatorId { get; set; }

        public int? EndingOperatorId { get; set; }

        public string Name { get; set; }

    }
}
