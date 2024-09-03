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
    /// of <see cref="BernoulliObservations"/> objects.
    /// </summary>
    [Combinator]
    [Description("Creates an observable sequence of BernoulliObservations objects.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    [JsonObject(MemberSerialization.OptIn)]
    public class BernoulliObservations : ObservationsModel
    {
        /// <summary>
        /// The logit P of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The logit P of the observations for each state.")]
        public double[,] LogitPs { get; set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(ObservationsModelTypeJsonConverter))]
        [Browsable(false)]
        public override ObservationsModelType ObservationsModelType => ObservationsModelType.Bernoulli;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get => [ LogitPs ];
        }

        /// <inheritdoc/>
        public BernoulliObservations () : base()
        {
        }

        /// <inheritdoc/>
        public BernoulliObservations (params object[] kwargs) : base(kwargs)
        {
        }

        /// <inheritdoc/>
        protected override void CheckParams(params object[] @params)
        {
            if (@params is not null && @params.Length != 1)
            {
                throw new ArgumentException($"The {nameof(BernoulliObservations)} operator requires exactly one parameter: {nameof(LogitPs)}.");
            }
        }

        /// <inheritdoc/>
        protected override void CheckKwargs(params object[] kwargs)
        {
            if (kwargs is null || kwargs.Length != 0)
            {
                throw new ArgumentException($"The {nameof(BernoulliObservations)} operator requires exactly zero constructor arguments.");
            }
        }

        /// <inheritdoc/>
        protected override void UpdateParams(params object[] @params)
        {
            LogitPs = (double[,])@params[0];
        }

        /// <summary>
        /// Returns an observable sequence of <see cref="BernoulliObservations"/> objects.
        /// </summary>
        public IObservable<BernoulliObservations> Process()
        {
            return Observable.Return(
                new BernoulliObservations {
                    Params = [ LogitPs ]
                });
        }

        /// <summary>
        /// Transforms an observable sequence of <see cref="PyObject"/> into an observable sequence 
        /// of <see cref="BernoulliObservations"/> objects by accessing internal attributes of the <see cref="PyObject"/>.
        /// </summary>
        public IObservable<BernoulliObservations> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var logitPsPyObj = (double[,])pyObject.GetArrayAttr("logit_ps");

                return new BernoulliObservations
                {
                    Params = [ logitPsPyObj ]
                };
            });
        }
    }
}