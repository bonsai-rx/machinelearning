using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents the result of a QR decomposition.
/// </summary>
public readonly struct QRDecompositionResult((Tensor q, Tensor r) result)
{
    /// <summary>
    /// Gets the orthogonal matrix Q.
    /// </summary>
    public Tensor Q => result.q;

    /// <summary>
    /// Gets the upper triangular matrix R.
    /// </summary>
    public Tensor R => result.r;
}
