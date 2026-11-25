using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents the result of solving of linear equations using the least squares method.
/// </summary>
public readonly struct LeastSquaresResult((
    Tensor solution,
    Tensor residuals,
    Tensor rank,
    Tensor singularValues
) result)
{
    /// <summary>
    /// Gets the solution to the system of equations.
    /// </summary>
    public Tensor Solution => result.solution;

    /// <summary>
    /// Gets the residual error.
    /// </summary>
    public Tensor Residuals => result.residuals;

    /// <summary>
    /// Gets the effective rank of the solution.
    /// </summary>
    public Tensor Rank => result.rank;

    /// <summary>
    /// Gets the singular values of the solution.
    /// </summary>
    public Tensor SingularValues => result.singularValues;
}
