using System.Collections.Generic;
using Newtonsoft.Json;
using System;

namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// Represents a stationary transitions model.
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    public class StationaryTransitions : TransitionsModel
    {
        /// <summary>
        /// The Log Ps of the transitions.
        /// </summary>
        public double[,] LogPs { get; private set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(TransitionsTypeJsonConverter))]
        public override TransitionsType TransitionsType => TransitionsType.Stationary;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get =>[ LogPs ];
            set
            {
                LogPs = (double[,])value[0];
                UpdateString();
            }
        }
    }
}