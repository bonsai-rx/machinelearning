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
        [JsonProperty]
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
        public override ObservationsModelType ObservationsModelType => ObservationsModelType.AutoRegressive;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get =>[ As, Bs, Vs, SqrtSigmas ];
            set
            {
                As = (double[,,])value[0];
                Bs = (double[,])value[1];
                Vs = (double[,,])value[2];
                SqrtSigmas = (double[,,])value[3];
                UpdateString();
            }
        }

        /// <inheritdoc/>
        [JsonProperty]
        public override Dictionary<string, object> Kwargs => new Dictionary<string, object>
        {
            ["lags"] = Lags,
        };

        /// <summary>
        /// Initializes a new instance of the <see cref="AutoRegressiveObservations"/> class.
        /// </summary>
        public AutoRegressiveObservations (params object[] args) : base(args)
        {
        }

        /// <inheritdoc/>
        protected override bool CheckConstructorArgs(params object[] args)
        {
            if (args is null || args.Length != 1)
            {
                throw new ArgumentException("The AutoRegressiveObservations operator requires a single argument specifying the number of lags.");
            }
            return true;
        }

        /// <inheritdoc/>
        protected override void UpdateKwargs(params object[] args)
        {
            Lags = args[0] switch
            {
                int lags => lags,
                var lags => Convert.ToInt32(lags),
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