using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Collections.Generic;
using Newtonsoft.Json;
using System.Xml.Serialization;
using Bonsai.ML.Python;

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
        [JsonProperty]
        public int[] HiddenLayerSizes { get; set; } = [50];

        /// <summary>
        /// The type of nonlinearity or activation function.
        /// </summary>
        public string Nonlinearity => "relu";

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
        public double[,,] Weights { get; set; } = null;

        /// <summary>
        /// The biases.
        /// </summary>
        [XmlIgnore]
        [Description("The biases.")]
        public double[,,] Biases { get; set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(TransitionsModelTypeJsonConverter))]
        public override TransitionsModelType TransitionsModelType => TransitionsModelType.NeuralNetworkRecurrent;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get => [LogPs, Weights, Biases];
            set
            {
                LogPs = (double[,])value[0];
                Weights = (double[,,])value[1];
                Biases = (double[,,])value[2];
                UpdateString();
            }
        }

        /// <inheritdoc/>
        [JsonProperty]
        public override Dictionary<string, object> Kwargs => new Dictionary<string, object>
        {
            ["hidden_layer_sizes"] = HiddenLayerSizes,
            ["nonlinearity"] = Nonlinearity,
        };

        /// <summary>
        /// Initializes a new instance of the <see cref="NeuralNetworkRecurrentTransitions"/> class.
        /// </summary>
        public NeuralNetworkRecurrentTransitions (params object[] args) : base(args)
        {
        }

        /// <inheritdoc/>
        protected override bool CheckConstructorArgs(params object[] args)
        {
            if (args is null || args.Length != 1)
            {
                throw new ArgumentException("The NeuralNetworkRecurrentTransitions operator requires a single argument specifying the hidden layer sizes.");
            }
            return true;
        }

        /// <inheritdoc/>
        protected override void UpdateKwargs(params object[] args)
        {
            HiddenLayerSizes = args[0] switch
            {
                int[] layers => layers,
                long[] layers => ConvertLongArrayToIntArray(layers),
                _ => null
            };
        }

        private static int[] ConvertLongArrayToIntArray(long[] longArray)
        {
            int count = longArray.Length;
            int[] intArray = new int[count];

            for (int i = 0; i < count; i++)
                intArray[i] = Convert.ToInt32(longArray[i]);

            return intArray;
        }

        /// <summary>
        /// Returns an observable sequence of <see cref="NeuralNetworkRecurrentTransitions"/> objects.
        /// </summary>
        public IObservable<NeuralNetworkRecurrentTransitions> Process()
        {
            return Observable.Return(
                new NeuralNetworkRecurrentTransitions(HiddenLayerSizes)
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
                var weightsPyObj = (double[,])pyObject.GetArrayAttr("weights");
                var biasesPyObj = (double[,])pyObject.GetArrayAttr("biases");

                return new NeuralNetworkRecurrentTransitions()
                {
                    Params = [logPsPyObj, weightsPyObj, biasesPyObj]
                };
            });
        }
    }
}