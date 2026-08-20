using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents the result of computing the sign and natural logarithm of the absolute value of the determinant.
/// </summary>
public readonly struct SignLogDeterminantResult((Tensor sign, Tensor logabsdet) result)
{
    /// <summary>
    /// The sign of the determinant.
    /// </summary>
    public Tensor Sign => result.sign;

    /// <summary>
    /// The natural logarithm of the absolute value of the determinant.
    /// </summary>
    public Tensor LogAbsDeterminant => result.logabsdet;
}
