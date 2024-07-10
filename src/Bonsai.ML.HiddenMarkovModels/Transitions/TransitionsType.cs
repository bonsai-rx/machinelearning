namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// Represents the type of transitions in a hidden Markov model.
    /// </summary>
    public enum TransitionsType
    {
        /// <summary>
        /// Stationary transitions.
        /// </summary>
        Stationary,

        /// <summary>
        /// Constrained stationary transitions.
        /// </summary>
        ConstrainedStationary,

        /// <summary>
        /// Sticky transitions.
        /// </summary>
        Sticky
    }
}