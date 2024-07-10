using System.ComponentModel;
using System.Text;
using Newtonsoft.Json;
using static Bonsai.ML.HiddenMarkovModels.Observations.ObservationsLookup;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    [JsonObject(MemberSerialization.OptIn)]
    public class ExponentialObservations : ObservationsModel
    {

        /// <summary>
        /// The log lambdas of the observations for each state.
        /// </summary>
        [Description("The log lambdas of the observations for each state.")]
        public double[,] LogLambdas { get; private set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
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