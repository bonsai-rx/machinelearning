using System.ComponentModel;
using Newtonsoft.Json;
using static Bonsai.ML.HiddenMarkovModels.Observations.ObservationsLookup;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    [JsonObject(MemberSerialization.OptIn)]
    public class BernoulliObservations : ObservationsModel
    {
        /// <summary>
        /// The logit P of the observations for each state.
        /// </summary>
        [Description("The logit P of the observations for each state.")]
        public double[,] LogitPs { get; private set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
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