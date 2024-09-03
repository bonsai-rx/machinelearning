using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Xml.Serialization;
using Newtonsoft.Json;
using Bonsai.ML.Python;

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
    public class StationaryTransitions : TransitionsModel
    {
        /// <summary>
        /// The Log Ps of the transitions.
        /// </summary>
        [XmlIgnore]
        [Description("The log Ps of the transitions.")]
        public double[,] LogPs { get; set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(TransitionsModelTypeJsonConverter))]
        [Browsable(false)]
        public override TransitionsModelType TransitionsModelType => TransitionsModelType.Stationary;

        /// <inheritdoc/>
        [JsonProperty]
          public override object[] Params
        {
            get => [LogPs];
        }

        /// <inheritdoc/>
        public StationaryTransitions() : base()
        {
        }

        /// <inheritdoc/>
        public StationaryTransitions(params object[] kwargs) : base(kwargs)
        {
        }

        /// <inheritdoc/>
        protected override void CheckParams(params object[] @params)
        {
            if (@params is not null && @params.Length != 1)
            {
                throw new ArgumentException($"The StickyTransitions operator requires exactly one parameter: {nameof(LogPs)}.");
            }
        }

        /// <inheritdoc/>
        protected override void UpdateParams(params object[] @params)
        {
            LogPs = (double[,])@params[0];
        }

        /// <inheritdoc/>
        protected override void CheckKwargs(params object[] kwargs)
        {
            if (kwargs is not null && kwargs.Length != 0)
            {
                throw new ArgumentException($"The StationaryTransitions operator requires zero constructor arguments.");
            }
        }

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