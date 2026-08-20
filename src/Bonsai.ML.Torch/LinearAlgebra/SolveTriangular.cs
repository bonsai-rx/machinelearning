using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes the solution to a triangular system of linear equations with a unique solution.
/// </summary>
[Combinator]
[Description("Computes the solution to a triangular system of linear equations with a unique solution.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class SolveTriangular
{
    /// <summary>
    /// Gets or sets whether the first matrix is upper triangular.
    /// </summary>
    /// <remarks>
    /// True indicates that the first matrix is upper triangular; otherwise, it is lower triangular.
    /// </remarks>
    [Description("True indicates that the first matrix is upper triangular; otherwise, it is lower triangular.")]
    public bool Upper { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to solve the system with the first matrix on the left or right (AX = B or XA = B).
    /// </summary>
    /// <remarks>
    /// True indicates that the system is solved with the first matrix on the left (AX = B); otherwise, it is solved
    /// with the first matrix on the right (XA = B).
    /// </remarks>
    [Description("True indicates that the system is solved with the first matrix on the left (AX = B); otherwise, it is solved with the first matrix on the right (XA = B).")]
    public bool Left { get; set; } = true;

    /// <summary>
    /// Gets or sets whether the first matrix has a unit diagonal, i.e., all diagonal elements are assumed to be 1.
    /// </summary>
    /// <remarks>
    /// True indicates that the first matrix has a unit diagonal, i.e., all diagonal elements are assumed to be 1;
    /// otherwise, the diagonal elements are used as-is.
    /// </remarks>
    [Description("True indicates that the first matrix has a unit diagonal, i.e., all diagonal elements are assumed to be 1; otherwise, the diagonal elements are used as-is.")]
    public bool UnitDiagonal { get; set; }

    /// <summary>
    /// Computes the solution to a triangular system of linear equations for each pair of input tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        return source.Select(value => solve_triangular(value.Item1, value.Item2, upper: Upper, left: Left, unitriangular: UnitDiagonal));
    }
}
