using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RUL_Prediction_NN.data_model
{
    public class Execution
    {
        [LoadColumn(0)]
        public int DefinitionId { get; set; }

        [LoadColumn(1)]
        public int ExecutionId { get; set; }

        [LoadColumn(2)]
        public DateTime StartDate { get; set; }

        [LoadColumn(3)]
        public DateTime EndDate { get; set; }

        [LoadColumn(4)]
        public int StartingOperatorId { get; set; }

        [LoadColumn(5)]
        public int EndingOperatorId { get; set; }

        [LoadColumn(6)]
        public string Name { get; set; }

    }
}
