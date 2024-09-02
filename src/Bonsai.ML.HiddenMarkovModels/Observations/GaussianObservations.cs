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
    /// of <see cref="GaussianObservations"/> objects.
    /// </summary>
    [Combinator]
    [Description("Creates an observable sequence of GaussianObservations objects.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    [JsonObject(MemberSerialization.OptIn)]
    public class GaussianObservations : ObservationsModel
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
        [JsonConverter(typeof(ObservationsModelTypeJsonConverter))]
        public override ObservationsModelType ObservationsModelType => ObservationsModelType.Gaussian;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get => [ Mus, SqrtSigmas ];
            set
            {
                Mus = (double[,])value[0];
                SqrtSigmas = (double[,,])value[1];
                UpdateString();
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