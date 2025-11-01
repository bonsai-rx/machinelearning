namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Specifies the training mode for a torch module.
/// </summary>
public enum TrainingMode
{
    /// <summary>
    /// Sets the model to training mode.
    /// </summary>
    Train,

    /// <summary>
    /// Sets the model to evaluation mode.
    /// </summary>
    Evaluation
}
