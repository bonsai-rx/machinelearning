using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Collections.Generic;
using Newtonsoft.Json;
using System.Xml.Serialization;
using Bonsai.ML.Python;
using System.Linq;

namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// Represents an operator that is used to create and transform an observable sequence
    /// of <see cref="NeuralNetworkRecurrentTransitions"/> objects.
    /// </summary>
    [Combinator]
    [Description("Creates an observable sequence of NeuralNetworkRecurrentTransitions objects.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    [JsonObject(MemberSerialization.OptIn)]
    public class NeuralNetworkRecurrentTransitions : TransitionsModel
    {
        /// <summary>
        /// The sizes of the hidden layers.
        /// </summary>
        [XmlIgnore]
        [Description("The sizes of the hidden layers.")]
        public int[] HiddenLayerSizes { get; set; } = [50];

        /// <summary>
        /// The type of nonlinearity or activation function.
        /// </summary>
        [Description("The type of nonlinearity or activation function.")]
        public NonlinearityType NonlinearityType { get; set; } = NonlinearityType.ReLU;

        /// <summary>
        /// The Log Ps of the transitions.
        /// </summary>
        [XmlIgnore]
        [Description("The log Ps of the transitions.")]
        public double[,] LogPs { get; set; } = null;

        /// <summary>
        /// The weights.
        /// </summary>
        [XmlIgnore]
        [Description("The weights.")]
        public List<double[,]> Weights { get; set; } = null;

        /// <summary>
        /// The biases.
        /// </summary>
        [XmlIgnore]
        [Description("The biases.")]
        public List<double[]> Biases { get; set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(TransitionsModelTypeJsonConverter))]
        public override TransitionsModelType TransitionsModelType => TransitionsModelType.NeuralNetworkRecurrent;

        private static readonly Dictionary<NonlinearityType, string> nonlinearityTypeLookup = new Dictionary<NonlinearityType, string>
        {
            { NonlinearityType.ReLU, "relu" },
            { NonlinearityType.Tanh, "tanh" },
            { NonlinearityType.Sigmoid, "sigmoid" }
        };

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get => [LogPs, Weights, Biases];
        }

        /// <inheritdoc/>
        [JsonProperty]
        [XmlIgnore]
        public new Dictionary<string, object> Kwargs => new Dictionary<string, object>
        {
            ["hidden_layer_sizes"] = HiddenLayerSizes,
            ["nonlinearity_type"] = nonlinearityTypeLookup[NonlinearityType],
        };

        /// <inheritdoc/>
        [XmlIgnore]
        public static new string[] KwargsArray => [ "hidden_layer_sizes", "nonlinearity_type" ];

        /// <inheritdoc/>
        public NeuralNetworkRecurrentTransitions () : base()
        {
        }

        /// <inheritdoc/>
        public NeuralNetworkRecurrentTransitions (params object[] args) : base(args)
        {
        }

        /// <inheritdoc/>
        protected override void CheckKwargs(params object[] kwargs)
        {
            if (kwargs is null || kwargs.Length != 2)
            {
                throw new ArgumentException($"The NeuralNetworkRecurrentTransitions operator requires exactly one constructor argument: {nameof(HiddenLayerSizes)}.");
            }
        }

        /// <inheritdoc/>
        protected override void UpdateKwargs(params object[] kwargs)
        {
            HiddenLayerSizes = kwargs[0] switch
            {
                int[] layers => layers,
                long[] layers => layers.Select(Convert.ToInt32).ToArray(),
                _ => null
            };
            try
            {
                NonlinearityType = (NonlinearityType)kwargs[1];
            }
            catch (InvalidCastException)
            {
                try
                {
                    NonlinearityType = nonlinearityTypeLookup.First(entry => entry.Value == (string)kwargs[1]).Key;
                }
                catch (KeyNotFoundException)
                {
                    throw new ArgumentException($"The NeuralNetworkRecurrentTransitions operator requires a valid nonlinearity type. The provided value was: {kwargs[1]} which is neither a valid NonlinearityType nor a valid string representation of a nonlinearity type.");
                }
            }

        }

        /// <inheritdoc/>
        protected override void CheckParams(params object[] @params)
        {
            if (@params is not null && @params.Length != 3)
            {
                throw new ArgumentException($"The NeuralNetworkRecurrentTransitions operator requires exactly three parameters: {nameof(LogPs)}, {nameof(Weights)}, and {nameof(Biases)}.");
            }
        }

        /// <inheritdoc/>
        protected override void UpdateParams(params object[] @params)
        {
            LogPs = @params[0] switch 
            {
                double[,] logPs => logPs,
                _ => null
            };

            Weights = @params[1] switch 
            {
                List<object> weights => weights.Select(weight => (double[,])weight).ToList(),
                _ => null
            };

            Biases = @params[2] switch 
            {
                List<object> biases => biases.Select(bias => (double[])bias).ToList(),
                _ => null
            };
        }

        /// <summary>
        /// Returns an observable sequence of <see cref="NeuralNetworkRecurrentTransitions"/> objects.
        /// </summary>
        public IObservable<NeuralNetworkRecurrentTransitions> Process()
        {
            return Observable.Return(
                new NeuralNetworkRecurrentTransitions([HiddenLayerSizes, NonlinearityType])
                {
                    Params = [LogPs, Weights, Biases]
                });
        }

        /// <summary>
        /// Transforms an observable sequence of <see cref="PyObject"/> into an observable sequence 
        /// of <see cref="NeuralNetworkRecurrentTransitions"/> objects by accessing internal attributes of the <see cref="PyObject"/>.
        /// </summary>
        public IObservable<NeuralNetworkRecurrentTransitions> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var logPsPyObj = (double[,])pyObject.GetArrayAttr("log_Ps");
                var weightsPyObj = (List<double[,]>)pyObject.GetArrayAttr("weights");
                var biasesPyObj = (List<double[]>)pyObject.GetArrayAttr("biases");
                var hiddenLayerSizesPyObj = (int[])pyObject.GetArrayAttr("hidden_layer_sizes");
                var nonlinearityTypePyObj = (string)pyObject.GetArrayAttr("nonlinearity_type");

                return new NeuralNetworkRecurrentTransitions([HiddenLayerSizes, NonlinearityType])
                {
                    Params = [logPsPyObj, weightsPyObj, biasesPyObj]
                };
            });
        }
    }
}