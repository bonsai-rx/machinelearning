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
    [JsonObject(MemberSerialization.OptIn)]
    public class ExponentialObservationsModel : ObservationsModelBuilder<ExponentialObservations>
    {

        /// <summary>
        /// The log lambdas of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The log lambdas of the observations for each state.")]
        public double[,] LogLambdas { get; private set; } = null;

        public IObservable<ExponentialObservations> Process()
        {
            return Observable.Return(
                new ExponentialObservations {
                    Params = [ LogLambdas ]
                });
        }

        public IObservable<ExponentialObservations> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var logLambdasPyObj = (double[,])pyObject.GetArrayAttr("log_lambdas");

                return new ExponentialObservations
                {
                    Params = [ logLambdasPyObj ]
                };
            });
        }
    }
}