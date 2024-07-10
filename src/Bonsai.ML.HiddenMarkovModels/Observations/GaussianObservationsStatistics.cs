using System.ComponentModel;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// Represents summary statistics and related data of an HMM model with gaussian observations.
    /// </summary>
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