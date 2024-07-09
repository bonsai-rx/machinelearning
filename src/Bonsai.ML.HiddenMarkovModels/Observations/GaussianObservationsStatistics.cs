using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    public class GaussianObservationsStatistics
    {
        /// <summary>
        /// The means of the observations for each state.
        /// </summary>
        [Description("The means of the observations for each state.")]
        public double[,] Means { get; set; }

        /// <summary>
        /// The standard deviations of the observations for each state.
        /// </summary>
        [Description("The standard deviations of the observations for each state.")]
        public double[,] StdDevs { get; set; }

        /// <summary>
        /// The covariance matrices of the observations for each state.
        /// </summary>
        [Description("The covariance matrices of the observations for each state.")]
        public double[,,] CovarianceMatrices { get; set; }

        /// <summary>
        /// The batch observations that the model has seen.
        /// </summary>
        [Description("The batch observations that the model has seen.")]
        public double[,] BatchObservations { get; set; }

        /// <summary>
        /// The sequence of inferred most probable states.
        /// </summary>
        [Description("The sequence of inferred most probable states.")]
        public long[] InferredMostProbableStates { get; set; }
    }
}