using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Xml.Serialization;
using Newtonsoft.Json;

namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// Represents an operator that is used to create and transform an observable sequence
    /// of <see cref="StationaryTransitions"/> objects.
    /// </summary>
    [Combinator]
    [Description("Creates an observable sequence of StationaryTransitions objects.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    [JsonObject(MemberSerialization.OptIn)]
    public class StationaryTransitionsModel
    {
        /// <summary>
        /// The Log Ps of the transitions.
        /// </summary>
        [XmlIgnore]
        [Description("The log Ps of the transitions.")]
        public double[,] LogPs { get; set; } = null;

        /// <summary>
        /// Returns an observable sequence of <see cref="StationaryTransitions"/> objects.
        /// </summary>
        public IObservable<StationaryTransitions> Process()
        {
            return Observable.Return(
                new StationaryTransitions
                {
                    Params = [LogPs]
                });
        }

        /// <summary>
        /// Transforms an observable sequence of <see cref="PyObject"/> into an observable sequence 
        /// of <see cref="StationaryTransitions"/> objects by accessing internal attributes of the <see cref="PyObject"/>.
        /// </summary>
        public IObservable<StationaryTransitions> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var logPsPyObj = (double[,])pyObject.GetArrayAttr("log_Ps");

                return new StationaryTransitions
                {
                    Params = [logPsPyObj]
                };
            });
        }
    }
}