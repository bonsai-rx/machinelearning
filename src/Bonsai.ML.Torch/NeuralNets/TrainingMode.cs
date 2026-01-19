namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Specifies the training mode for a torch module.
/// </summary>
public enum TrainingMode
{
    /// <summary>
    /// Sets the module to evaluation mode.
    /// </summary>
    Evaluation = 0,

    /// <summary>
    /// Sets the module to training mode.
    /// </summary>
    Train = 1,
}
