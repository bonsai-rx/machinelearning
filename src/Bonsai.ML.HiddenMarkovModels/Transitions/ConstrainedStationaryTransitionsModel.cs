using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using Newtonsoft.Json;

namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// Represents an operator that is used to create and transform an observable sequence
    /// of <see cref="ConstrainedStationaryTransitions"/> objects.
    /// </summary>
    [Combinator]
    [Description("Creates an observable sequence of ConstrainedStationaryTransitions objects.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    [JsonObject(MemberSerialization.OptIn)]
    public class ConstrainedStationaryTransitionsModel : StationaryTransitionsModel
    {
        private int[,] transitionMask = null;
        private string transitionMaskString = null;

        /// <summary>
        /// The mask which gets applied to the transition matrix to prohibit certain transitions. 
        /// It must be written in JSON format as an int[,] with the same shape as the transition matrix (nStates x nStates). 
        /// For example, the mask [[1, 0], [1, 1]] is valid and would prohibit transitions from state 0 to state 1.
        /// </summary>
        [Description("The mask which gets applied to the transition matrix to prohibit certain transitions. It must be written in JSON format as an int[,] with the same shape as the transition matrix (nStates x nStates). For example, the mask [[1, 0], [1, 1]] is valid and would prohibit transitions from state 0 to state 1.")]
        [JsonProperty]
        public string TransitionMask
        {
            get => transitionMaskString;
            set
            {
                try
                {
                    transitionMask = (int[,])NumpyHelper.NumpyParser.ParseString(value, typeof(int));
                    transitionMaskString = value;
                }
                finally { }
            }
        }

        /// <summary>
        /// Returns an observable sequence of <see cref="ConstrainedStationaryTransitions"/> objects.
        /// </summary>
        new public IObservable<ConstrainedStationaryTransitions> Process()
        {
            return Observable.Return(
                new ConstrainedStationaryTransitions(transitionMask)
                {
                    Params = [LogPs]
                });
        }

        /// <summary>
        /// Transforms an observable sequence of <see cref="PyObject"/> into an observable sequence 
        /// of <see cref="ConstrainedStationaryTransitions"/> objects by accessing internal attributes of the <see cref="PyObject"/>.
        /// </summary>
        new public IObservable<ConstrainedStationaryTransitions> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var logPsPyObj = (double[,])pyObject.GetArrayAttr("log_Ps");
                var transitionMaskPyObj = (int[,])pyObject.GetArrayAttr("transition_mask");

                return new ConstrainedStationaryTransitions(transitionMaskPyObj)
                {
                    Params = [logPsPyObj]
                };
            });
        }
    }
}