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
    public class AutoRegressiveObservations : Observations<AutoRegressiveObservations>
    {

        /// <summary>
        /// The lags of the observations for each state.
        /// </summary>
        [Description("The lags of the observations for each state.")]
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

        public AutoRegressiveObservations()
        {
        }

        public AutoRegressiveObservations(int lags)
        {
            Lags = lags;
        }

        public AutoRegressiveObservations(params object[] args)
        {
            Lags = (int)args[0];
        }

        [JsonProperty]
        public override object[] Params
        {
            get { return new object[] { As, Bs, Vs, SqrtSigmas }; }
            set
            {
                As = (double[,,])value[0];
                Bs = (double[,])value[1];
                Vs = (double[,,])value[2];
                SqrtSigmas = (double[,,])value[3];
            }
        }

        public IObservable<AutoRegressiveObservations> Process()
        {
            return Observable.Return(
                new AutoRegressiveObservations(lags: Lags) {
                    As = As,
                    Bs = Bs,
                    Vs = Vs,
                    SqrtSigmas = SqrtSigmas
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

                return new AutoRegressiveObservations(lagsPyObj)
                {
                    As = asPyObj,
                    Bs = bsPyObj,
                    Vs = vsPyObj,
                    SqrtSigmas = sqrtSigmasPyObj
                };
            });
        }

        public override string ToString()
        {
            if (As is null || Bs is null || Vs is null || SqrtSigmas is null) return $"observation_params=None,observation_kwargs={{'lags':{Lags}}}";
            return $"observation_params=({NumpyHelper.NumpyParser.ParseArray(As)},{NumpyHelper.NumpyParser.ParseArray(Bs)},{NumpyHelper.NumpyParser.ParseArray(Vs)},{NumpyHelper.NumpyParser.ParseArray(SqrtSigmas)}),observation_kwargs={{'lags':{Lags}}}";
        }
    }
}