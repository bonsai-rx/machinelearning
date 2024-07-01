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
    public class BernoulliObservations : ObservationParams<BernoulliObservations>
    {
        /// <summary>
        /// The logit P of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("logit_ps")]
        [Description("The logit P of the observations for each state.")]
        public double[,] LogitPs { get; private set; }

        public override object[] Params
        {
            get { return new object[] { LogitPs }; }
            set { LogitPs = (double[,])value[0]; }
        }

        public override IObservable<BernoulliObservations> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var logitPsPyObj = (double[,])pyObject.GetArrayAttr("logit_ps");

                return new BernoulliObservations
                {
                    LogitPs = logitPsPyObj
                };
            });
        }

        public override string ToString()
        {
            return $"observation_params=({NumpyHelper.NumpyParser.ParseArray(LogitPs)},)";
        }
    }
}