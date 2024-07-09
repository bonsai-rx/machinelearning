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
    public class AutoRegressiveObservationsModel : ObservationsModelBuilder<AutoRegressiveObservations>
    {
        /// <summary>
        /// The lags of the observations for each state.
        /// </summary>
        [Description("The lags of the observations for each state.")]
        [JsonProperty]
        public int Lags { get; set; } = 1;

        /// <summary>
        /// The As of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The As of the observations for each state.")]
        public double[,,] As { get; private set; } = null;

        /// <summary>
        /// The bs of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The bs of the observations for each state.")]
        public double[,] Bs { get; private set; } = null;

        /// <summary>
        /// The Vs of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The Vs of the observations for each state.")]
        public double[,,] Vs { get; private set; } = null;

        /// <summary>
        /// The square root sigmas of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The square root sigmas of the observations for each state.")]
        public double[,,] SqrtSigmas { get; private set; } = null;

        public IObservable<AutoRegressiveObservations> Process()
        {
            return Observable.Return(
                new AutoRegressiveObservations (Lags) {
                    Params = [ As, Bs, Vs, SqrtSigmas ],
                });
        }

        public IObservable<AutoRegressiveObservations> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var lagsPyObj = (int)pyObject.GetArrayAttr("lags");
                var asPyObj = (double[,,])pyObject.GetArrayAttr("As");
                var bsPyObj = (double[,])pyObject.GetArrayAttr("bs");
                var vsPyObj = (double[,,])pyObject.GetArrayAttr("Vs");
                var sqrtSigmasPyObj = (double[,,])pyObject.GetArrayAttr("_sqrt_Sigmas");

                return new AutoRegressiveObservations(Lags)
                {
                    Params = [
                        asPyObj,
                        bsPyObj,
                        vsPyObj,
                        sqrtSigmasPyObj
                    ]
                };
            });
        }
    }
}