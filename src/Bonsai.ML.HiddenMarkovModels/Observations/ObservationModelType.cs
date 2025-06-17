namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// Represents the type of observations in a hidden Markov model.
    /// </summary>
    public enum ObservationModelType
    {
        /// <summary>
        /// Gaussian observations.
        /// </summary>
        Gaussian,

        /// <summary>
        /// Exponential observations.
        /// </summary>
        Exponential,

        /// <summary>
        /// Bernoulli observations.
        /// </summary>
        Bernoulli,

        /// <summary>
        /// Poisson observations.
        /// </summary>
        Poisson,

        /// <summary>
        /// Autoregressive observations.
        /// </summary>
        AutoRegressive,

        /// <summary>
        /// Categorical observations.
        /// </summary>
        Categorical
    }
}