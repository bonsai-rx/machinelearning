using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Xml.Serialization;
using Newtonsoft.Json;
using System.Linq;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    [JsonObject(MemberSerialization.OptIn)]
    public class GaussianObservationsModel : ObservationsModelBuilder<GaussianObservations>
    {

        /// <summary>
        /// The means of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The means of the observations for each state.")]
        public double[,] Mus { get; private set; } = null;

        /// <summary>
        /// The standard deviations of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The standard deviations of the observations for each state.")]
        public double[,,] SqrtSigmas { get; private set; } = null;

        public IObservable<GaussianObservations> Process()
        {
            return Observable.Return(
                new GaussianObservations {
                    Params = [ Mus, SqrtSigmas ]
                });
        }

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