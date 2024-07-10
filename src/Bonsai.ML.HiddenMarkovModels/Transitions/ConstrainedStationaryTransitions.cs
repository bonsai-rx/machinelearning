using System;
using System.Collections.Generic;
using Newtonsoft.Json;

namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// Represents a constrained stationary transitions model.
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    public class ConstrainedStationaryTransitions : StationaryTransitions
    {
        private int[,] transitionMask = null;

        /// <summary>
        /// The transition mask.
        /// </summary>
        public int[,] TransitionMask { get => transitionMask; set { transitionMask = value; UpdateString(); } }

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(TransitionsTypeJsonConverter))]
        public override TransitionsType TransitionsType => TransitionsType.ConstrainedStationary;

        /// <inheritdoc/>
        [JsonProperty]
        public override Dictionary<string, object> Kwargs => new Dictionary<string, object>
        {
            ["transition_mask"] = TransitionMask,
        };

        /// <summary>
        /// Initializes a new instance of the <see cref="ConstrainedStationaryTransitions"/> class.
        /// </summary>
        public ConstrainedStationaryTransitions(params object[] args)
        {
            if (args is not null && args.Length == 1)
            {
                TransitionMask = args[0] switch
                {
                    int[,] mask => mask,
                    long[,] mask => ConvertLongArrayToIntArray(mask),
                    _ => null
                };
                UpdateString();
            }
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
    }
}