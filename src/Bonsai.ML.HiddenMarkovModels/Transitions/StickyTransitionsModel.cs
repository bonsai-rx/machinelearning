using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using Newtonsoft.Json;

namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// Represents an operator that is used to create and transform an observable sequence
    /// of <see cref="StickyTransitions"/> objects.
    /// </summary>
    [Combinator]
    [Description("Creates an observable sequence of StickyTransitions objects.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    [JsonObject(MemberSerialization.OptIn)]
    public class StickyTransitionsModel : StationaryTransitionsModel
    {
        /// <summary>
        /// The alpha parameter.
        /// </summary>
        [Description("The alpha parameter.")]
        [JsonProperty]
        public double Alpha { get; set; } = 1.0;

        /// <summary>
        /// The kappa parameter.
        /// </summary>
        [Description("The kappa parameter.")]
        [JsonProperty]
        public double Kappa { get; set; } = 100.0;

        /// <summary>
        /// Returns an observable sequence of <see cref="StickyTransitions"/> objects.
        /// </summary>
        new public IObservable<StickyTransitions> Process()
        {
            return Observable.Return(new StickyTransitions([Alpha, Kappa]));
        }

        /// <summary>
        /// Transforms an observable sequence of <see cref="PyObject"/> into an observable sequence 
        /// of <see cref="StickyTransitions"/> objects by accessing internal attributes of the <see cref="PyObject"/>.
        /// </summary>
        new public IObservable<StickyTransitions> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var alphaPyObj = (int[,])pyObject.GetArrayAttr("alpha");
                var kappaPyObj = (int[,])pyObject.GetArrayAttr("kappa");

                return new StickyTransitions([alphaPyObj, kappaPyObj]);
            });
        }
    }
}