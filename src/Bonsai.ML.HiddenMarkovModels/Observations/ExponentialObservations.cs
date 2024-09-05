using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Xml.Serialization;
using Newtonsoft.Json;
using Bonsai.ML.Python;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// Represents an operator that is used to create and transform an observable sequence
    /// of <see cref="ExponentialObservations"/> objects.
    /// </summary>
    [Combinator]
    [Description("Creates an observable sequence of ExponentialObservations objects.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    [JsonObject(MemberSerialization.OptIn)]
    public class ExponentialObservations : ObservationsModel
    {

        /// <summary>
        /// The log lambdas of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The log lambdas of the observations for each state.")]
        public double[,] LogLambdas { get; set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(ObservationsModelTypeJsonConverter))]
        [Browsable(false)]
        public override ObservationsModelType ObservationsModelType => ObservationsModelType.Exponential;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get => [ LogLambdas ];
        }

        /// <inheritdoc/>
        public ExponentialObservations () : base()
        {
        }

        /// <inheritdoc/>
        public ExponentialObservations (params object[] kwargs) : base(kwargs)
        {
        }

        /// <inheritdoc/>
        protected override void CheckParams(params object[] @params)
        {
            if (@params is not null && @params.Length != 1)
            {
                throw new ArgumentException($"The {nameof(ExponentialObservations)} operator requires exactly one parameter: {nameof(LogLambdas)}.");
            }
        }

        /// <inheritdoc/>
        protected override void UpdateParams(params object[] @params)
        {
            LogLambdas = (double[,])@params[0];
        }

        /// <inheritdoc/>
        protected override void CheckKwargs(params object[] kwargs)
        {
            if (kwargs is null || kwargs.Length != 0)
            {
                throw new ArgumentException($"The {nameof(ExponentialObservations)} operator requires exactly zero constructor arguments.");
            }
        }

        /// <summary>
        /// Returns an observable sequence of <see cref="ExponentialObservations"/> objects.
        /// </summary>
        public IObservable<ExponentialObservations> Process()
        {
            return Observable.Return(
                new ExponentialObservations {
                    Params = [ LogLambdas ]
                });
        }

        /// <summary>
        /// Transforms an observable sequence of <see cref="PyObject"/> into an observable sequence 
        /// of <see cref="ExponentialObservations"/> objects by accessing internal attributes of the <see cref="PyObject"/>.
        /// </summary>
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