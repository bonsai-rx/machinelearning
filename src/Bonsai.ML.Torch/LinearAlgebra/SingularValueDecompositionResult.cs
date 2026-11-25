using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents the result of a singular value decomposition.
/// </summary>
public readonly struct SingularValueDecompositionResult((
    Tensor u,
    Tensor s,
    Tensor vh
) result)
{
    /// <summary>
    /// Gets the unitary matrix with left singular vectors as columns.
    /// </summary>
    public Tensor U => result.u;

    /// <summary>
    /// Gets the singular values.
    /// </summary>
    public Tensor S => result.s;

    /// <summary>
    /// Gets the unitary matrix with right singular vectors as rows.
    /// </summary>
    public Tensor Vh => result.vh;
}
