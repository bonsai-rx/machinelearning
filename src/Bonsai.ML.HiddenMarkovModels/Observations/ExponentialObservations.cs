using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Xml.Serialization;
using Newtonsoft.Json;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ExponentialObservations : ObservationParams<ExponentialObservations>
    {

        /// <summary>
        /// The log lambdas of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("log_lambdas")]
        [Description("The log lambdas of the observations for each state.")]
        public double[,] LogLambdas { get; private set; }

        public override object[] Params
        {
            get { return new object[] { LogLambdas }; }
            set
            {
                LogLambdas = (double[,])value[0];
            }
        }

        public override IObservable<ExponentialObservations> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var logLambdasPyObj = (double[,])pyObject.GetArrayAttr("log_lambdas");

                return new ExponentialObservations
                {
                    LogLambdas = logLambdasPyObj
                };
            });
        }
        public override string ToString()
        {
            return $"observation_params=({NumpyHelper.NumpyParser.ParseArray(LogLambdas)},)";
        }
    }
}