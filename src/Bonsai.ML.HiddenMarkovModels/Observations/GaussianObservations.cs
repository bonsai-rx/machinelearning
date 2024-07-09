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
    public class GaussianObservations : Observations<GaussianObservations>
    {

        /// <summary>
        /// The means of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The means of the observations for each state.")]
        public double[,] Mus { get; private set; }

        /// <summary>
        /// The standard deviations of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The standard deviations of the observations for each state.")]
        public double[,,] SqrtSigmas { get; private set; }

        [JsonProperty]
        public override object[] Params
        {
            get { return new object[] { Mus, SqrtSigmas }; }
            set
            {
                Mus = (double[,])value[0];
                SqrtSigmas = (double[,,])value[1];
            }
        }

        public IObservable<GaussianObservations> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var musPyObj = (double[,])pyObject.GetArrayAttr("mus");
                var sigmasPyObj = (double[,,])pyObject.GetArrayAttr("_sqrt_Sigmas");

                return new GaussianObservations
                {
                    Mus = musPyObj,
                    SqrtSigmas = sigmasPyObj
                };
            });
        }

        public override string ToString()
        {
            return $"observation_params=({NumpyHelper.NumpyParser.ParseArray(Mus)}, {NumpyHelper.NumpyParser.ParseArray(SqrtSigmas)})";
        }
    }
}