using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Xml.Serialization;
using Newtonsoft.Json;

namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// Represents a constrained stationary transitions model.
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    public class NeuralNetworkRecurrentTransitions : TransitionsModel
    {
        private int[] hiddenLayerSizes = [50];
        private readonly string nonlinearity = "relu";

        /// <summary>
        /// The sizes of the hidden layers.
        /// </summary>
        [XmlIgnore]
        public int[] HiddenLayerSizes { get => hiddenLayerSizes; set {hiddenLayerSizes = value; UpdateString();} }

        /// <summary>
        /// The sizes of the hidden layers.
        /// </summary>
        public string Nonlinearity => nonlinearity;

        /// <summary>
        /// The Log Ps of the transitions.
        /// </summary>
        public double[,] LogPs { get; private set; } = null;

        /// <summary>
        /// The weights of the transitions.
        /// </summary>
        public double[,,] Weights { get; private set; } = null;

        /// <summary>
        /// The biases of the transitions.
        /// </summary>
        public double[,,] Biases { get; private set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(TransitionsTypeJsonConverter))]
        public override TransitionsType TransitionsType => TransitionsType.NeuralNetworkRecurrent;

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
        public NeuralNetworkRecurrentTransitions(params object[] args)
        {
            if (args is not null && args.Length > 0)
            {
                HiddenLayerSizes = args[0] switch
                {
                    int[] layers => layers,
                    long[] layers => ConvertLongArrayToIntArray(layers),
                    _ => null
                };
                UpdateString();
            }
        }

        private static int[] ConvertLongArrayToIntArray(long[] longArray)
        {
            int count = longArray.Length;
            int[] intArray = new int[count];

            for (int i = 0; i < count; i++)
                intArray[i] = Convert.ToInt32(longArray[i]);

            return intArray;
        }
    }
}