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
    /// The unitary matrix with left singular vectors as columns.
    /// </summary>
    public Tensor U => result.u;

    /// <summary>
    /// The singular values.
    /// </summary>
    public Tensor S => result.s;

    /// <summary>
    /// The unitary matrix with right singular vectors as rows.
    /// </summary>
    public Tensor Vh => result.vh;
}
