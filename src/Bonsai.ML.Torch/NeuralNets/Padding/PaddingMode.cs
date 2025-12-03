namespace Bonsai.ML.Torch.NeuralNets.Padding;

/// <summary>
/// Specifies the padding mode for a neural network module.
/// </summary>
public enum PaddingMode
{
    /// <summary>
    /// Zero padding.
    /// </summary>
    Zero,

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
    Replication,
}