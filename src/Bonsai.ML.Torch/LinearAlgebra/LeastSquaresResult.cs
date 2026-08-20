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
    /// The solution to the system of equations.
    /// </summary>
    public Tensor Solution => result.solution;

    /// <summary>
    /// The residual error.
    /// </summary>
    public Tensor Residuals => result.residuals;

    /// <summary>
    /// The effective rank of the solution.
    /// </summary>
    public Tensor Rank => result.rank;

    /// <summary>
    /// The singular values of the solution.
    /// </summary>
    public Tensor SingularValues => result.singularValues;
}
