using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes the solution to the least squares and least norm problems for a full rank matrix A of size m×n and a matrix B of size m×k.
/// </summary>
[Combinator]
[Description("Computes the solution to the system tensordot(A, X) = B.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class LeastSquaresSolve
{
    /// <summary>
    /// Computes the solution to the least squares and least norm problems for a full rank matrix A of size m×n and a matrix B of size m×k.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<LeastSquaresResult> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        return source.Select(value =>
        {
            var (solution, residuals, rank, singularValues) = linalg.lstsq(value.Item1, value.Item2);
            return new LeastSquaresResult(
                solution,
                residuals,
                rank,
                singularValues);
        });
    }

    /// <summary>
    /// Represents the result of solving of linear equations using the least squares method.
    /// </summary>
    /// <param name="solution"></param>
    /// <param name="residuals"></param>
    /// <param name="rank"></param>
    /// <param name="singularValues"></param>
    public readonly struct LeastSquaresResult(
        Tensor solution,
        Tensor residuals,
        Tensor rank,
        Tensor singularValues
    )
    {
        /// <summary>
        /// The solution to the system of equations.
        /// </summary>
        public Tensor Solution => solution;

        /// <summary>
        /// The residual error.
        /// </summary>
        public Tensor Residuals => residuals;

        /// <summary>
        /// The effective rank of the solution.
        /// </summary>
        public Tensor Rank => rank;

        /// <summary>
        /// The singular values of the solution.
        /// </summary>
        public Tensor SingularValues => singularValues;
    }
}