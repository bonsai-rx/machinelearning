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
    public class PoissonObservationsModel : ObservationsModelBuilder<PoissonObservations>
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

        /// <inheritdoc/>
        public override string ToString()
        {
            if (LogLambdas is null) 
                return $"observation_params=None";

            return $"observation_params=({NumpyHelper.NumpyParser.ParseArray(LogLambdas)},)";
        }
    }
}