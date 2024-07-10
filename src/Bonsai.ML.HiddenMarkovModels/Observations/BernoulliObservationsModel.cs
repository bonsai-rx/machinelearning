using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Xml.Serialization;
using Newtonsoft.Json;

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
    public class BernoulliObservationsModel
    {
        /// <summary>
        /// The logit P of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The logit P of the observations for each state.")]
        public double[,] LogitPs { get; set; } = null;

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