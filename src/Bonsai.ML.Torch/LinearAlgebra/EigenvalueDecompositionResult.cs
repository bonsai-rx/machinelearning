using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents the result of an eigenvalue decomposition.
/// </summary>
/// <param name="result">The tuple containing the eigenvalues and eigenvectors.</param>
public readonly struct EigenvalueDecompositionResult((Tensor eigenvalues, Tensor eigenvectors) result)
{
    /// <summary>
    /// Gets the eigenvalues of the decomposition.
    /// </summary>
    public Tensor Eigenvalues => result.eigenvalues;

    /// <summary>
    /// Gets the eigenvectors of the decomposition.
    /// </summary>
    public Tensor Eigenvectors => result.eigenvectors;
}
