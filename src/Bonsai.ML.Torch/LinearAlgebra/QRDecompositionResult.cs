using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents the result of a QR decomposition.
/// </summary>
public readonly struct QRDecompositionResult((Tensor q, Tensor r) result)
{
    /// <summary>
    /// The orthogonal matrix Q.
    /// </summary>
    public Tensor Q => result.q;

    /// <summary>
    /// The upper triangular matrix R.
    /// </summary>
    public Tensor R => result.r;
}
