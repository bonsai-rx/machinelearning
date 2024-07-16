using System.Collections.Generic;
using Newtonsoft.Json;
using System;

namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// Represents a sticky transitions model.
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    public class StickyTransitions : StationaryTransitions
    {
        private double alpha = 1.0;
        private double kappa = 100.0;

        /// <summary>
        /// The alpha parameter.
        /// </summary>
        public double Alpha { get => alpha; set {alpha = value; UpdateString(); } }

        /// <summary>
        /// The kappa parameter.
        /// </summary>
        public double Kappa { get => kappa; set {kappa = value; UpdateString(); } }

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(TransitionsTypeJsonConverter))]
        public override TransitionsType TransitionsType => TransitionsType.Sticky;

        /// <inheritdoc/>
        [JsonProperty]
        public override Dictionary<string, object> Kwargs => new Dictionary<string, object>
        {
            ["alpha"] = Alpha,
            ["kappa"] = Kappa,
        };

        /// <summary>
        /// Initializes a new instance of the <see cref="StickyTransitions"/> class.
        /// </summary>
        public StickyTransitions (params object[] args)
        {
            if (args is not null && args.Length == 2)
            {
                Alpha = args[0] switch
                {
                    double a => a,
                    var a => Convert.ToDouble(a)
                };
                Kappa = args[1] switch
                {
                    double k => k,
                    var k => Convert.ToDouble(k)
                };
                UpdateString();
            }
        }
    }
}