using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Collections.Generic;
using System.Xml.Serialization;
using Newtonsoft.Json;
using Bonsai.ML.Python;

namespace Bonsai.ML.Hmm.Python.Transitions
{
    /// <summary>
    /// Represents an operator that is used to create and transform an observable sequence
    /// of <see cref="StickyTransitions"/> objects.
    /// </summary>
    [Combinator]
    [Description("Creates an observable sequence of StickyTransitions objects.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    [JsonObject(MemberSerialization.OptIn)]
    public class StickyTransitions : TransitionModel
    {
        /// <summary>
        /// The alpha parameter.
        /// </summary>
        [Description("The alpha parameter.")]
        public double Alpha { get; set; } = 1.0;

        /// <summary>
        /// The kappa parameter.
        /// </summary>
        [Description("The kappa parameter.")]
        public double Kappa { get; set; } = 100.0;

        /// <summary>
        /// The Log Ps of the transitions.
        /// </summary>
        [XmlIgnore]
        [Description("The log Ps of the transitions.")]
        public double[,] LogPs { get; set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(TransitionModelTypeJsonConverter))]
        [Browsable(false)]
        public override TransitionModelType TransitionModelType => TransitionModelType.Sticky;

        /// <inheritdoc/>
        [JsonProperty]
        [Browsable(false)]
        public override object[] Params
        {
            get => [LogPs];
        }

        /// <inheritdoc/>
        [JsonProperty]
        [XmlIgnore]
        public override Dictionary<string, object> Kwargs => new Dictionary<string, object>
        {
            ["alpha"] = Alpha,
            ["kappa"] = Kappa,
        };

        /// <inheritdoc/>
        [XmlIgnore]
        public static new string[] KwargsArray => [ "alpha", "kappa" ];

        /// <inheritdoc/>
        public StickyTransitions() : base()
        {
        }

        /// <inheritdoc/>
        public StickyTransitions(params object[] kwargs) : base(kwargs)
        {
        }

        /// <inheritdoc/>
        protected override void CheckKwargs(params object[] kwargs)
        {
            if (kwargs is not null && kwargs.Length != 2)
            {
                throw new ArgumentException($"The StickyTransitions operator requires exactly two constructor arguments: {nameof(Alpha)} and {nameof(Kappa)}.");
            }
        }

        /// <inheritdoc/>
        protected override void UpdateKwargs(params object[] kwargs)
        {
            Alpha = kwargs[0] switch
            {
                double a => a,
                var a => Convert.ToDouble(a)
            };
            Kappa = kwargs[1] switch
            {
                double k => k,
                var k => Convert.ToDouble(k)
            };
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
            if (@params is not null)
            {
                LogPs = (double[,])@params[0];
            }
        }

        /// <summary>
        /// Returns an observable sequence of <see cref="StickyTransitions"/> objects.
        /// </summary>
        public IObservable<StickyTransitions> Process()
        {
            return Observable.Return(new StickyTransitions([Alpha, Kappa]) 
            {
                Params = [LogPs]
            });
        }

        /// <summary>
        /// Transforms an observable sequence of <see cref="PyObject"/> into an observable sequence 
        /// of <see cref="StickyTransitions"/> objects by accessing internal attributes of the <see cref="PyObject"/>.
        /// </summary>
        public IObservable<StickyTransitions> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var alphaPyObj = (int[,])pyObject.GetArrayAttr("alpha");
                var kappaPyObj = (int[,])pyObject.GetArrayAttr("kappa");
                var logPsPyObj = (double[,])pyObject.GetArrayAttr("log_Ps");

                return new StickyTransitions([alphaPyObj, kappaPyObj]) 
                {
                    Params = [logPsPyObj]
                };
            });
        }
    }
}