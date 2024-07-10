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
    public class BernoulliObservationsModel
    {
        /// <summary>
        /// The logit P of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The logit P of the observations for each state.")]
        public double[,] LogitPs { get; private set; } = null;

        public IObservable<BernoulliObservations> Process()
        {
            return Observable.Return(
                new BernoulliObservations {
                    Params = [ LogitPs ]
                });
        }

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