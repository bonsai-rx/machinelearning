using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Xml.Serialization;
using Newtonsoft.Json;
using System.Linq;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    [JsonObject(MemberSerialization.OptIn)]
    public class GaussianObservations : ObservationsModel
    {

        /// <summary>
        /// The means of the observations for each state.
        /// </summary>
        [Description("The means of the observations for each state.")]
        public double[,] Mus { get; private set; } = null;

        /// <summary>
        /// The standard deviations of the observations for each state.
        /// </summary>
        [Description("The standard deviations of the observations for each state.")]
        public double[,,] SqrtSigmas { get; private set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        public override ObservationsType ObservationsType => ObservationsType.Gaussian;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get { return [ Mus, SqrtSigmas ]; }
            set
            {
                Mus = (double[,])value[0];
                SqrtSigmas = (double[,,])value[1];
            }
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            if (Mus is null || SqrtSigmas is null) 
                return $"observation_params=None";

            return $"observation_params=({NumpyHelper.NumpyParser.ParseArray(Mus)}," +
                $"{NumpyHelper.NumpyParser.ParseArray(SqrtSigmas)})";
        }
    }
}