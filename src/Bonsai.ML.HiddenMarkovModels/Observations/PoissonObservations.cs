using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Xml.Serialization;
using Newtonsoft.Json;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    [JsonObject(MemberSerialization.OptIn)]
    public class PoissonObservations : ObservationsModel
    {

        /// <summary>
        /// The log lambdas of the observations for each state.
        /// </summary>
        [Description("The log lambdas of the observations for each state.")]
        public double[,] LogLambdas { get; private set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        public override ObservationsType ObservationsType => ObservationsType.Poisson;

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