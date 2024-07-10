using System.Collections.Generic;
using Newtonsoft.Json;
using System;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// Represents a categorical observations model.
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    public class CategoricalObservations : ObservationsModel
    {
        private int categories = 2;

        /// <summary>
        /// The number of categories in the observations.
        /// </summary>
        [JsonProperty]
        public int Categories { get => categories; set {categories = value; UpdateString(); } }

        /// <summary>
        /// The logit of the observations for each state.
        /// </summary>
        public double[,,] Logits { get; private set; } = null;


        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(ObservationsTypeJsonConverter))]
        public override ObservationsType ObservationsType => ObservationsType.Categorical;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get =>[ Logits ];
            set
            {
                Logits = (double[,,])value[0];
                UpdateString();
            }
        }

        /// <inheritdoc/>
        [JsonProperty]
        public override Dictionary<string, object> Kwargs => new Dictionary<string, object>
        {
            ["C"] = categories,
        };

        /// <summary>
        /// Initializes a new instance of the <see cref="CategoricalObservations"/> class.
        /// </summary>
        public CategoricalObservations (params object[] args)
        {
            categories = Convert.ToInt32(args[0]);
            UpdateString();
        }
    }
}