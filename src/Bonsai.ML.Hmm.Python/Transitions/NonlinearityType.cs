namespace Bonsai.ML.Hmm.Python.Transitions
{
    /// <summary>
    /// Represents the type of nonlinearity to use in a recurrent neural network.
    /// </summary>
    public enum NonlinearityType
    {
        /// <summary>
        /// Rectified linear unit (ReLU) nonlinearity.
        /// </summary>
        ReLU,

        /// <summary>
        /// Tanh nonlinearity.
        /// </summary>
        Tanh,

        /// <summary>
        /// Sigmoid nonlinearity.
        /// </summary>
        Sigmoid
    }
}