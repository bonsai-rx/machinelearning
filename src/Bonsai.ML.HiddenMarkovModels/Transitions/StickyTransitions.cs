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
    /// of <see cref="StickyTransitions"/> objects.
    /// </summary>
    [Combinator]
    [Description("Creates an observable sequence of StickyTransitions objects.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    [JsonObject(MemberSerialization.OptIn)]
    public class StickyTransitions : TransitionsModel
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
        /// The Log Ps of the transitions.
        /// </summary>
        [XmlIgnore]
        [Description("The log Ps of the transitions.")]
        public double[,] LogPs { get; set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(TransitionsModelTypeJsonConverter))]
        public override TransitionsModelType TransitionsModelType => TransitionsModelType.Sticky;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get => [LogPs];
            set
            {
                LogPs = (double[,])value[0];
                UpdateString();
            }
        }

        /// <inheritdoc/>
        [JsonProperty]
        public override Dictionary<string, object> Kwargs => new Dictionary<string, object>
        {
            ["alpha"] = Alpha,
            ["kappa"] = Kappa,
        };

        /// <inheritdoc/>
        public StickyTransitions(params object[] args) : base(args)
        {
        }

        /// <inheritdoc/>
        protected override bool CheckConstructorArgs(params object[] args)
        {
            if (args is not null && args.Length != 2)
            {
                throw new ArgumentException("The StickyTransitions operator requires two constructor arguments specifying the alpha and kappa parameters.");
            }
            return true;
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

        /// <summary>
        /// Returns an observable sequence of <see cref="StickyTransitions"/> objects.
        /// </summary>
        public IObservable<StickyTransitions> Process()
        {
            return Observable.Return(
                new StickyTransitions([Alpha, Kappa]) {
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

                return new StickyTransitions([alphaPyObj, kappaPyObj]) {
                    Params = [logPsPyObj]
                };
            });
        }
    }
}