namespace Bonsai.ML.Torch.NeuralNets.Module;

/// <summary>
/// Specifies the padding mode for a neural network module.
/// </summary>
public enum PaddingMode
{
    /// <summary>
    /// Constant padding.
    /// </summary>
    Constant,

    /// <summary>
    /// Reflection padding.
    /// </summary>
    Reflection,

    /// <summary>
    /// Replication padding.
    /// </summary>
    Replication
}