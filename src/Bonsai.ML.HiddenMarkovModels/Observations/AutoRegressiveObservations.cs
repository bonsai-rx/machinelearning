using System;
using System.Collections.Generic;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Xml.Serialization;
using Newtonsoft.Json;
using Bonsai.ML.Python;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// Represents an operator that is used to create and transform an observable sequence
    /// of <see cref="AutoRegressiveObservations"/> objects.
    /// </summary>
    [Combinator]
    [Description("Creates an observable sequence of AutoRegressiveObservations objects.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    [JsonObject(MemberSerialization.OptIn)]
    public class AutoRegressiveObservations : ObservationsModel
    {
        /// <summary>
        /// The lags of the observations for each state.
        /// </summary>
        [Description("The lags of the observations for each state.")]
        public int Lags { get; set; } = 1;

        /// <summary>
        /// The As of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The As of the observations for each state.")]
        public double[,,] As { get; set; } = null;

        /// <summary>
        /// The bs of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The bs of the observations for each state.")]
        public double[,] Bs { get; set; } = null;

        /// <summary>
        /// The Vs of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The Vs of the observations for each state.")]
        public double[,,] Vs { get; set; } = null;

        /// <summary>
        /// The square root sigmas of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The square root sigmas of the observations for each state.")]
        public double[,,] SqrtSigmas { get; set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(ObservationsModelTypeJsonConverter))]
        [Browsable(false)]
        public override ObservationsModelType ObservationsModelType => ObservationsModelType.AutoRegressive;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get =>[ As, Bs, Vs, SqrtSigmas ];
        }

        /// <inheritdoc/>
        [JsonProperty]
        [XmlIgnore]
        public override Dictionary<string, object> Kwargs => new Dictionary<string, object>
        {
            ["lags"] = Lags,
        };

        /// <inheritdoc/>
        [XmlIgnore]
        public static new string[] KwargsArray => [ "lags" ];

        /// <inheritdoc/>
        public AutoRegressiveObservations () : base()
        {
        }

        /// <inheritdoc/>
        public AutoRegressiveObservations (params object[] args) : base(args)
        {
        }

        /// <inheritdoc/>
        protected override void CheckKwargs(params object[] kwargs)
        {
            if (kwargs is null || kwargs.Length != 1)
            {
                throw new ArgumentException($"The AutoRegressiveObservations operator requires exactly one constructor argument: {nameof(Lags)}.");
            }
        }

        /// <inheritdoc/>
        protected override void UpdateKwargs(params object[] args)
        {
            Lags = args[0] switch
            {
                int lags => lags,
                var lags => Convert.ToInt32(lags)
            };
        }

        /// <inheritdoc/>
        protected override void CheckParams(params object[] @params)
        {
            if (@params is not null && @params.Length != 4)
            {
                throw new ArgumentException($"The {nameof(AutoRegressiveObservations)} operator requires exactly four parameters: {nameof(As)}, {nameof(Bs)}, {nameof(Vs)}, and {nameof(SqrtSigmas)}.");
            }
        }

        /// <inheritdoc/>
        protected override void UpdateParams(params object[] @params)
        {
            As = @params[0] switch 
            {
                double[,,] As => As,
                _ => null
            };

            Bs = @params[1] switch 
            {
                double[,] Bs => Bs,
                _ => null
            };

            Vs = @params[2] switch 
            {
                double[,,] Vs => Vs,
                _ => null
            };

            SqrtSigmas = @params[3] switch 
            {
                double[,,] SqrtSigmas => SqrtSigmas,
                _ => null
            };
        }

        /// <summary>
        /// Returns an observable sequence of <see cref="AutoRegressiveObservations"/> objects.
        /// </summary>
        public IObservable<AutoRegressiveObservations> Process()
        {
            return Observable.Return(
                new AutoRegressiveObservations (Lags) {
                    Params = [ As, Bs, Vs, SqrtSigmas ],
                });
        }

        /// <summary>
        /// Transforms an observable sequence of <see cref="PyObject"/> into an observable sequence 
        /// of <see cref="AutoRegressiveObservations"/> objects by accessing internal attributes of the <see cref="PyObject"/>.
        /// </summary>
        public IObservable<AutoRegressiveObservations> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var lagsPyObj = (int)pyObject.GetArrayAttr("lags");
                var asPyObj = (double[,,])pyObject.GetArrayAttr("As");
                var bsPyObj = (double[,])pyObject.GetArrayAttr("bs");
                var vsPyObj = (double[,,])pyObject.GetArrayAttr("Vs");
                var sqrtSigmasPyObj = (double[,,])pyObject.GetArrayAttr("_sqrt_Sigmas");

                return new AutoRegressiveObservations(Lags)
                {
                    Params = [
                        asPyObj,
                        bsPyObj,
                        vsPyObj,
                        sqrtSigmasPyObj
                    ]
                };
            });
        }
    }
}