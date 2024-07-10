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
    [WorkflowElementCategory(ElementCategory.Source)]
    [JsonObject(MemberSerialization.OptIn)]
    public class PoissonObservationsModel
    {

        /// <summary>
        /// The log lambdas of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The log lambdas of the observations for each state.")]
        public double[,] LogLambdas { get; private set; } = null;

        public IObservable<PoissonObservations> Process()
        {
            return Observable.Return(
                new PoissonObservations {
                    Params = [ LogLambdas ]
                });
        }

        public IObservable<PoissonObservations> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var logLambdasPyObj = (double[,])pyObject.GetArrayAttr("log_lambdas");

                return new PoissonObservations {
                    Params = [ logLambdasPyObj ]
                };
            });
        }
    }
}