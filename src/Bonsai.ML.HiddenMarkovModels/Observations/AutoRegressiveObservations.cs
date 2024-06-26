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
    public class AutoRegressiveObservations : ObservationParams<AutoRegressiveObservations>
    {

        /// <summary>
        /// The lags of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("lags")]
        [Description("The lags of the observations for each state.")]
        public int Lags { get; private set; }

        /// <summary>
        /// The As of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("As")]
        [Description("The As of the observations for each state.")]
        public double[,] As { get; private set; }

        /// <summary>
        /// The bs of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("bs")]
        [Description("The bs of the observations for each state.")]
        public double[,] Bs { get; private set; }

        /// <summary>
        /// The Vs of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("Vs")]
        [Description("The Vs of the observations for each state.")]
        public double[,,] Vs { get; private set; }

        /// <summary>
        /// The square root sigmas of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("_sqrt_Sigmas")]
        [Description("The square root sigmas of the observations for each state.")]
        public double[,,] SqrtSigmas { get; private set; }

        public AutoRegressiveObservations(int lags)
        {
            Lags = lags;
        }

        public override object[] Params
        {
            get { return new object[] { As, Bs, Vs, SqrtSigmas }; }
            set
            {
                As = (double[,])value[0];
                Bs = (double[,])value[1];
                Vs = (double[,,])value[2];
                SqrtSigmas = (double[,,])value[3];
            }
        }

        public override IObservable<AutoRegressiveObservations> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var lagsPyObj = (int)pyObject.GetArrayAttr("lags");
                var asPyObj = (double[,])pyObject.GetArrayAttr("As");
                var bsPyObj = (double[,])pyObject.GetArrayAttr("bs");
                var vsPyObj = (double[,,])pyObject.GetArrayAttr("Vs");
                var sqrtSigmasPyObj = (double[,,])pyObject.GetArrayAttr("_sqrt_Sigmas");

                return new AutoRegressiveObservations(lagsPyObj)
                {
                    As = asPyObj,
                    Bs = bsPyObj,
                    Vs = vsPyObj,
                    SqrtSigmas = sqrtSigmasPyObj
                };
            });
        }
    }
}