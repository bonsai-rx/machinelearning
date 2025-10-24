namespace Bonsai.ML.Hmm.Python.Transitions
{
    /// <summary>
    /// Represents the type of transitions in a hidden Markov model.
    /// </summary>
    public enum TransitionModelType
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
        Sticky,

        /// <summary>
        /// Neural network recurrent transitions.
        /// </summary>
        NeuralNetworkRecurrent
    }
}