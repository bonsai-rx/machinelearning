using Newtonsoft.Json;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// Represents an exponential observations model.
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    public class ExponentialObservations : ObservationsModel
    {
        /// <summary>
        /// The log lambdas of the observations for each state.
        /// </summary>
        public double[,] LogLambdas { get; private set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(ObservationsTypeJsonConverter))]
        public override ObservationsType ObservationsType => ObservationsType.Exponential;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get { return [ LogLambdas ]; }
            set
            {
                LogLambdas = (double[,])value[0];
                UpdateString();
            }
        }
    }
}