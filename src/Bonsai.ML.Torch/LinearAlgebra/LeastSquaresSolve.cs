using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes the solution to the least squares and least norm problems for a full rank matrix A of size m*n and a matrix B of size m*k.
/// </summary>
[Combinator]
[Description("Computes the solution to the least squares and least norm problems for a full rank matrix A of size m*n and a matrix B of size m*k.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class LeastSquaresSolve
{
    /// <summary>
    /// Computes the solution to the least squares and least norm problems for a full rank matrix A of size m*n and a matrix B of size m*k.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<LeastSquaresResult> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        return source.Select(value => new LeastSquaresResult(linalg.lstsq(value.Item1, value.Item2)));
    }

    /// <summary>
    /// Represents the result of solving of linear equations using the least squares method.
    /// </summary>
    /// <param name="result"></param>
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
}
