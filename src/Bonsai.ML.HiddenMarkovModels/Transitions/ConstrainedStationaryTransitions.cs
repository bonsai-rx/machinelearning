using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Collections.Generic;
using System.Xml.Serialization;
using Newtonsoft.Json;
using Bonsai.ML.Python;

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
    public class ConstrainedStationaryTransitions : TransitionsModel
    {
        private int[,] transitionMask = null;

        /// <summary>
        /// The mask which gets applied to the transition matrix to prohibit certain transitions. 
        /// It must be written in JSON format as an int[,] with the same shape as the transition matrix (nStates x nStates). 
        /// For example, the mask [[1, 0], [1, 1]] is valid and would prohibit transitions from state 0 to state 1.
        /// </summary>
        [Description("The mask which gets applied to the transition matrix to prohibit certain transitions. It must be written in JSON format as an int[,] with the same shape as the transition matrix (nStates x nStates). For example, the mask [[1, 0], [1, 1]] is valid and would prohibit transitions from state 0 to state 1.")]
        [JsonProperty]
        public string TransitionMask
        {
            get => transitionMask != null ? NumpyHelper.NumpyParser.ParseArray(transitionMask) : "";
            set => transitionMask = (int[,])NumpyHelper.NumpyParser.ParseString(value, typeof(int));
        }

        /// <summary>
        /// The Log Ps of the transitions.
        /// </summary>
        [XmlIgnore]
        [Description("The log Ps of the transitions.")]
        public double[,] LogPs { get; set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(TransitionsModelTypeJsonConverter))]
        public override TransitionsModelType TransitionsModelType => TransitionsModelType.ConstrainedStationary;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get => [LogPs];
        }

        /// <inheritdoc/>
        [JsonProperty]
        [XmlIgnore]
        public new Dictionary<string, object> Kwargs => new Dictionary<string, object>
        {
            ["transition_mask"] = TransitionMask,
        };

        /// <inheritdoc/>
        [XmlIgnore]
        public static new string[] KwargsArray => [ "transition_mask" ];

        /// <inheritdoc/>
        public ConstrainedStationaryTransitions() : base()
        {
        }

        /// <inheritdoc/>
        public ConstrainedStationaryTransitions (params object[] kwargs) : base(kwargs)
        {
        }

        /// <inheritdoc/>
        protected override void CheckKwargs(params object[] kwargs)
        {
            if (kwargs is null || kwargs.Length != 1)
            {
                throw new ArgumentException($"The ConstrainedStationaryTransitions operator requires exactly one keyword argument: {nameof(transitionMask)}.");
            }
        }

        /// <inheritdoc/>
        protected override void UpdateKwargs(params object[] kwargs)
        {
            transitionMask = kwargs[0] switch
            {
                int[,] mask => mask,
                long[,] mask => ConvertLongArrayToIntArray(mask),
                _ => null
            };
        }

        private static int[,] ConvertLongArrayToIntArray(long[,] longArray)
        {
            int rows = longArray.GetLength(0);
            int cols = longArray.GetLength(1);
            int[,] intArray = new int[rows, cols];

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    intArray[i, j] = Convert.ToInt32(longArray[i, j]);

            return intArray;
        }

        /// <inheritdoc/>
        protected override void CheckParams(params object[] @params)
        {
            if (@params is not null && @params.Length != 1)
            {
                throw new ArgumentException($"The ConstrainedStationaryTransitions operator requires exactly one parameter: {nameof(LogPs)}.");
            }
        }

        /// <inheritdoc/>
        protected override void UpdateParams(params object[] @params)
        {
            LogPs = @params[0] switch {
                double[,] logPs => logPs,
                _ => null
            };
        }

        /// <summary>
        /// Returns an observable sequence of <see cref="ConstrainedStationaryTransitions"/> objects.
        /// </summary>
        public IObservable<ConstrainedStationaryTransitions> Process()
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
        public IObservable<ConstrainedStationaryTransitions> Process(IObservable<PyObject> source)
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