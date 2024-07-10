using Newtonsoft.Json;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// Represents an bernoulli observations model.
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    public class BernoulliObservations : ObservationsModel
    {
        /// <summary>
        /// The logit P of the observations for each state.
        /// </summary>
        public double[,] LogitPs { get; private set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(ObservationsTypeJsonConverter))]
        public override ObservationsType ObservationsType => ObservationsType.Bernoulli;

        /// <inheritdoc/>
        [JsonProperty]
         public override object[] Params
        {
            get { return [ LogitPs ]; }
            set 
            { 
                LogitPs = (double[,])value[0];
                UpdateString();
            }
        }
    }
}