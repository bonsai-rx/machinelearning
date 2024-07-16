using Newtonsoft.Json;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// Represents a gaussian observations model.
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    public class GaussianObservations : ObservationsModel
    {

        /// <summary>
        /// The means of the observations for each state.
        /// </summary>
        public double[,] Mus { get; private set; } = null;

        /// <summary>
        /// The standard deviations of the observations for each state.
        /// </summary>
        public double[,,] SqrtSigmas { get; private set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(ObservationsTypeJsonConverter))]
        public override ObservationsType ObservationsType => ObservationsType.Gaussian;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get => [ Mus, SqrtSigmas ];
            set
            {
                Mus = (double[,])value[0];
                SqrtSigmas = (double[,,])value[1];
                UpdateString();
            }
        }
    }
}