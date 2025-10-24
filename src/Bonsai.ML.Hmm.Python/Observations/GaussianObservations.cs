using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Xml.Serialization;
using Newtonsoft.Json;
using Bonsai.ML.Python;

namespace Bonsai.ML.Hmm.Python.Observations
{
    /// <summary>
    /// Represents an operator that is used to create and transform an observable sequence
    /// of <see cref="GaussianObservations"/> objects.
    /// </summary>
    [Combinator]
    [Description("Creates an observable sequence of GaussianObservations objects.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    [JsonObject(MemberSerialization.OptIn)]
    public class GaussianObservations : ObservationModel
    {
        /// <summary>
        /// The means of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The means of the observations for each state.")]
        public double[,] Mus { get; set; } = null;

        /// <summary>
        /// The standard deviations of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The standard deviations of the observations for each state.")]
        public double[,,] SqrtSigmas { get; set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(ObservationModelTypeJsonConverter))]
        [Browsable(false)]
        public override ObservationModelType ObservationModelType => ObservationModelType.Gaussian;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get => [ Mus, SqrtSigmas ];
        }

        /// <inheritdoc/>
        public GaussianObservations () : base()
        {
        }

        /// <inheritdoc/>
        public GaussianObservations (params object[] kwargs) : base(kwargs)
        {
        }

        /// <inheritdoc/>
        protected override void CheckParams(params object[] @params)
        {
            if (@params is not null && @params.Length != 2)
            {
                throw new ArgumentException($"The {nameof(GaussianObservations)} operator requires exactly two parameters: {nameof(Mus)} and {nameof(SqrtSigmas)}.");
            }
        }

        /// <inheritdoc/>
        protected override void UpdateParams(params object[] @params)
        {
            Mus = (double[,])@params[0];
            SqrtSigmas = (double[,,])@params[1];
        }

        /// <inheritdoc/>
        protected override void CheckKwargs(params object[] kwargs)
        {
            if (kwargs is null || kwargs.Length != 0)
            {
                throw new ArgumentException($"The {nameof(GaussianObservations)} operator requires exactly zero constructor arguments.");
            }
        }

        /// <summary>
        /// Returns an observable sequence of <see cref="GaussianObservations"/> objects.
        /// </summary>
        public IObservable<GaussianObservations> Process()
        {
            return Observable.Return(
                new GaussianObservations {
                    Params = [ Mus, SqrtSigmas ]
                });
        }

        /// <summary>
        /// Transforms an observable sequence of <see cref="PyObject"/> into an observable sequence 
        /// of <see cref="GaussianObservations"/> objects by accessing internal attributes of the <see cref="PyObject"/>.
        /// </summary>
        public IObservable<GaussianObservations> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var musPyObj = (double[,])pyObject.GetArrayAttr("mus");
                var sqrtSigmasPyObj = (double[,,])pyObject.GetArrayAttr("_sqrt_Sigmas");

                return new GaussianObservations {
                    Params = [ Mus, SqrtSigmas ]
                };
            });
        }
    }
}