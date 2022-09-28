using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Tensorflow.NumPy;
using Accord.Math;
using RUL_Prediction_NN.data_model;

namespace RUL_Prediction_NN.Data
{
    public static class pd
    {

        /*
         * General read and write functions
         */

        public static void to_csv(string path, string sep = ",", List<string> headers = null, NDArray columns = null, bool append = true, TypeCode type = TypeCode.Empty)
        {
            
            var writer = new StreamWriter(path, append: append);    //Configuracion del objeto de escritura

            var culture = new CultureInfo("en-US");

            // Write headers 
            if (headers is not null)
            {
                foreach (var a in headers)
                {
                    writer.Write(a + ",");
                }

                //Cambio de fila
                writer.WriteLine();
            }


            // Write values 
            if (columns is not null)
            {
                // Si el array es unidimensional escribirlo en una fila
                if (columns.ndim > 1 && columns.shape[0] == 1)
                {
                    for (int i = 0; i < columns.shape[1]; i++)
                    {
                        if (type == TypeCode.Empty)
                        {
                            writer.Write("{0}" + sep, ((float)columns[0][i]).ToString("0.00", culture));
                        }

                        else if (type == TypeCode.Int32)
                        {
                            writer.Write("{0}" + sep, ((int)columns[0][i]).ToString("0.00", culture));
                        }

                        else if (type == TypeCode.Double)
                        {
                            writer.Write("{0}" + sep, ((double)columns[0][i]).ToString("0.00", culture));
                        }

                        else if (type == TypeCode.String)
                        {
                            writer.Write("{0}" + sep, ((string)columns[0][i]).ToString(culture));
                        }

                    }

                    writer.WriteLine();
                }

                // Si el NDArray es multidimensional escribir cada fila en una linea
                else if (columns.ndim > 1 && columns.shape[0] > 1)
                {
                    for (int i = 0; i < columns.shape[0]; i++)
                    {
                        for (int j = 0; j < columns.shape[1]; j++)
                        {
                            if (type == TypeCode.Empty)
                            {
                                writer.Write("{0,2}", ((float)columns[i][j]).ToString("0.00", culture));
                                if (j != columns.shape[1] - 1)
                                {
                                    writer.Write(sep);
                                }
                            }

                            else if (type == TypeCode.Int32)
                            {
                                writer.Write("{0,2}", ((int)columns[i][j]).ToString("0.00", culture));
                                if (j != columns.shape[1] - 1)
                                {
                                    writer.Write(sep);
                                }
                            }

                            else if (type == TypeCode.Double)
                            {
                                writer.Write("{0,2}", ((double)columns[i][j]).ToString("0.00", culture));
                                if (j != columns.shape[1] - 1)
                                {
                                    writer.Write(sep);
                                }
                            }

                        }

                        writer.WriteLine();

                    }
                    //Console.WriteLine("CSV Saved");

                }

                // Si el NDArray es un elemento 
                else
                {
                    for (int k = 0; k < columns.shape[0]; k++)
                    {
                        if (type == TypeCode.Empty)
                        {
                            writer.Write("{0,2}" + sep, ((float)columns[k]).ToString("0.00", culture));
                        }

                        else if (type == TypeCode.Int32)
                        {
                            writer.Write("{0,2}" + sep, ((int)columns[k]).ToString("0.00", culture));
                        }

                        else if (type == TypeCode.Double)
                        {
                            writer.Write("{0,2}" + sep, ((double)columns[k]).ToString("0.00", culture));
                        }

                    }


                    writer.WriteLine();
                }

            }


            writer.Close();

        }

        public static float[,] read_csv(string path, string sep = ",", bool headers = false)
        {


            var reader = new StreamReader(path);        //Configuracion del objeto de lectura

            var culture = new CultureInfo("en-US");     //Configuracion para leer tipo de dato double con separador decimal '.'

            var float_list = new List<float>();       //variable temporal para la conversion de string a double
            var data = new List<List<float>>();        //variable de retorno con los datos

            while (!reader.EndOfStream)
            {

                var line = reader.ReadLine();       //Lectura de cada linea
                var values = line.Split(sep);       //Separacion de los datos

                if (!headers)                       //Indica si la primera linea son encabezados para descartarla
                {

                    foreach (var v in values)
                    {
                        if (v != "")
                        {
                            //float_list.Add(Convert.ToDouble(v, culture));      //Conversion a double
                            float_list.Add(Convert.ToSingle(v, culture));
                        }

                    }

                    data.Add(new List<float>(float_list));

                    float_list.Clear();
                }

                else
                {
                    headers = false;
                }


            }

            reader.Close();                             //Cierre del documento

            return list_to_Array(data);                 //Devolver arreglo multidimensional de datos

        }

        private static float[,] list_to_Array(List<List<float>> list)
        {
            /*
            Para realizar el preprocesado de los datos se necesita
            que los datos sean un arreglo multidimensional
            Para ello se lleva de lista a array
            */

            var array = list.ToArray();

            float[][] data = new float[array.Length][];

            for (int i = 0; i < array.Length; i++)
            {
                data[i] = array[i].ToArray();
            }

            return data.ToMatrix();

        }










        /*
         * Read and write functions for RO data
         */


        public static List<Phase> read_phases(string path, bool headers = true, char sep = ',')
        {

            /*
             * Lee objetos fases del set de datos de RO
             */

            var reader = new StreamReader(path);        //Configuracion del objeto de lectura

            var phase_list = new List<Phase>();

            int executionId = 0;
            int temp = 0;
            int phaseId = 0;

            while (!reader.EndOfStream)
            {

                var line = reader.ReadLine();       //Lectura de cada linea
                var values = line.Split(sep);       //Separacion de los datos

                if (!headers)                       //Indica si la primera linea son encabezados para descartarla
                {
                    //executionId = Convert.ToInt32(values[1]);
                    if (values.Length < 4)
                    {
                        break;
                    }

                    //if (Convert.ToInt32(values[1]) > temp)
                    //{
                    //    executionId = executionId + 1;
                    //    temp = Convert.ToInt32(values[1]);
                    //}
                    executionId = Convert.ToInt32(values[1]);

                    phase_list.Add(new Phase
                    {
                        PhaseId = phaseId,
                        EntityId = Convert.ToInt32(values[0]),
                        ExecutionId = executionId,
                        Time = Convert.ToDateTime(values[2]),
                        Text = values[3],
                    });

                    phaseId = phaseId + 1;

                }

                else
                {
                    headers = false;
                }


            }

            reader.Close();                             //Cierre del documento

            return phase_list;

        }

        public static List<Alarm> read_alarms(string path, bool headers = true, char sep = ',')
        {

            /*
             * Lee objetos alarmas del set de datos de RO
             */


            var reader = new StreamReader(path);        //Configuracion del objeto de lectura

            var alarm_list = new List<Alarm>();

            while (!reader.EndOfStream)
            {

                var line = reader.ReadLine();       //Lectura de cada linea
                var values = line.Split(sep);       //Separacion de los datos

                if (!headers)                       //Indica si la primera linea son encabezados para descartarla
                {

                    if (values.Length < 5)
                    {
                        break;
                    }


                    alarm_list.Add(new Alarm
                    {
                        EntityId = Convert.ToInt32(values[0]),
                        DeviceId = Convert.ToInt32(values[1]),
                        Time = Convert.ToDateTime(values[2]),
                        Text = values[3],
                        Discriminator = values[4]
                    });


                }

                else
                {
                    headers = false;
                }


            }

            reader.Close();                             //Cierre del documento

            return alarm_list;

        }

        public static List<Sample> read_samples(string path, bool headers = true, char sep = ',', bool partial = false)
        {
            /*
             * Lee objetos de muestras del set de datos de RO
             * Si está activa la variable 'partial' se leen solo algunos campos del objeto muestras
             */

            var reader = new StreamReader(path);        //Configuracion del objeto de lectura

            var sample_list = new List<Sample>();

            int timeId = 0;
            DateTime temp_time = DateTime.MinValue;

            while (!reader.EndOfStream)
            {

                var line = reader.ReadLine();       //Lectura de cada linea
                var values = line.Split(sep);       //Separacion de los datos

                if (!headers)                       //Indica si la primera linea son encabezados para descartarla
                {

                    if (!partial)
                    {
                        if (values.Length < 4)
                        {
                            break;
                        }

                        if (Convert.ToDateTime(values[0]) > temp_time)
                        {
                            temp_time = Convert.ToDateTime(values[0]);
                            timeId = timeId + 1;
                        }

                        sample_list.Add(new Sample
                        {
                            TimeId = timeId,
                            Time = Convert.ToDateTime(values[0]),
                            Value = Convert.ToDouble(values[1]),
                            VariableId = Convert.ToInt32(values[2]),
                            Name = values[3]
                        }); ;
                    }

                    else
                    {
                        sample_list.Add(new Sample
                        {
                            VariableId = Convert.ToInt32(values[0]),
                            Time = Convert.ToDateTime(values[1]),
                            Value = Convert.ToDouble(values[2]),
                        });
                    }

                }

                else
                {
                    headers = false;
                }


            }

            reader.Close();                             //Cierre del documento

            return sample_list;

        }

        public static void samples_to_csv(string path, string sep = ",", List<Sample> columns = null, bool append = true)
        {

            /*
             * Guarda muestras en fichero .csv
             */

            var writer = new StreamWriter(path, append: append);    //Configuracion del objeto de escritura

            var culture = new CultureInfo("en-US");

            foreach (var s in columns)
            {
                writer.Write("{0}", (s.VariableId).ToString(culture));
                writer.Write(sep);
                writer.Write(s.Time);
                writer.Write(sep);
                writer.Write("{0}", (s.Value).ToString("0.0000", culture));
                writer.WriteLine();
            }


            writer.Close();

        }

        public static void phases_to_csv(string path, string sep = ",", List<Phase> columns = null, bool append = true)
        {
            var writer = new StreamWriter(path, append: append);

            var culture = new CultureInfo("en-US");

            foreach(var s in columns)
            {
                writer.Write("{0}", s.EntityId.ToString(culture));
                writer.Write(sep);
                writer.Write("{0}", s.ExecutionId.ToString(culture));
                writer.Write(sep);
                writer.Write(s.Time);
                writer.Write(sep);
                writer.Write("{0}", s.Text);
                writer.WriteLine();
            }
            writer.Close();
        }

        public static List<Execution> read_executions(string path, bool headers = true, char sep = ',')
        {

            /*
             * Lee objetos ejecuciones del set de datos de RO
             */
            
            var reader = new StreamReader(path);        //Configuracion del objeto de lectura

            var excecution_list = new List<Execution>();

            int executionId = 0;
            int temp = 0;

            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();       //Lectura de cada linea
                var values = line.Split(sep);       //Separacion de los datos

                if (!headers)                       //Indica si la primera linea son encabezados para descartarla
                {
                    //executionId = Convert.ToInt32(values[1]);
                    if (values.Length < 7)
                    {
                        break;
                    }
                    
                    //if (Convert.ToInt32(values[1]) > temp)
                    //{
                    //    executionId = executionId + 1;
                    //    temp = Convert.ToInt32(values[1]);
                    //}
                    executionId = Convert.ToInt32(values[1]);

                    if(values[4] == "NULL" && values[5] == "NULL")
                    {
                        excecution_list.Add(new Execution
                        {
                            DefinitionId = Convert.ToInt32(values[0]),
                            ExecutionId = executionId,
                            StartDate = Convert.ToDateTime(values[2]),
                            EndDate = Convert.ToDateTime(values[3]),
                            StartingOperatorId = null,
                            EndingOperatorId = null,
                            Name = values[6]
                        });
                        
                    }

                    else if (values[5] == "NULL")
                    {
                        /*
                        var tempExecution = new Execution();
                        tempExecution.DefinitionId = Convert.ToInt32(values[0]);
                        tempExecution.ExecutionId = executionId;
                        tempExecution.StartDate = Convert.ToDateTime(values[2]);
                        tempExecution.EndDate = Convert.ToDateTime(values[3]);
                        tempExecution.StartingOperatorId = Convert.ToInt32(values[4]);
                        tempExecution.EndingOperatorId = null;
                        tempExecution.Name = values[6];
                        */
                        excecution_list.Add(new Execution
                        {
                            DefinitionId = Convert.ToInt32(values[0]),
                            ExecutionId = executionId,
                            StartDate = Convert.ToDateTime(values[2]),
                            EndDate = Convert.ToDateTime(values[3]),
                            StartingOperatorId = Convert.ToInt32(values[4]),
                            EndingOperatorId = null,
                            Name = values[6]
                        });

                    }
                    else if(values[4] == "NULL")
                    {
                        excecution_list.Add(new Execution
                        {
                            DefinitionId = Convert.ToInt32(values[0]),
                            ExecutionId = executionId,
                            StartDate = Convert.ToDateTime(values[2]),
                            EndDate = Convert.ToDateTime(values[3]),
                            StartingOperatorId = null,
                            EndingOperatorId = Convert.ToInt32(values[5]),
                            Name = values[6]
                        });
                    }

                    else
                    {
                        excecution_list.Add(new Execution
                        {
                            DefinitionId = Convert.ToInt32(values[0]),
                            ExecutionId = executionId,
                            StartDate = Convert.ToDateTime(values[2]),
                            EndDate = Convert.ToDateTime(values[3]),
                            StartingOperatorId = Convert.ToInt32(values[4]),
                            EndingOperatorId = Convert.ToInt32(values[5]),
                            Name = values[6]
                        });
                    }

                }

                else
                {
                    headers = false;
                }

            }

            reader.Close();                             //Cierre del documento

            return excecution_list;

        }

        public static List<List<double>> read_samples_of_executions(string path, bool headers = false, string sep = ",")
        {

            /*
             * Lee muestras de ejecuciones
             */

            var reader = new StreamReader(path);        //Configuracion del objeto de lectura

            var culture = new CultureInfo("en-US");     //Configuracion para leer tipo de dato double con separador decimal '.'

            var temp_list = new List<double>();       //variable temporal para la conversion de string a double
            var data = new List<List<double>>();        //variable de retorno con los datos

            while (!reader.EndOfStream)
            {

                var line = reader.ReadLine();       //Lectura de cada linea
                var values = line.Split(sep);       //Separacion de los datos

                if (!headers)                       //Indica si la primera linea son encabezados para descartarla
                {

                    foreach (var v in values)
                    {
                        if (v != "")
                        {
                            temp_list.Add(Convert.ToDouble(v, culture));      //Conversion a double
                        }

                    }

                    data.Add(new List<double>(temp_list));

                    temp_list.Clear();
                }

                else
                {
                    headers = false;
                }


            }

            reader.Close();                             //Cierre del documento

            return (data);

        }










    }
}
