using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Xml.Serialization;
using Newtonsoft.Json;

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
            set { LogitPs = (double[,])value[0]; }
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            if (LogitPs is null) 
                return $"observation_params=None";

            return $"observation_params=({NumpyHelper.NumpyParser.ParseArray(LogitPs)},)";
        }
    }
}