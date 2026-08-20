using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes the numerical rank of a matrix.
/// </summary>
[Combinator]
[Description("Computes the numerical rank of a matrix.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class MatrixRank
{
    /// <summary>
    /// Gets or sets the absolute tolerance for singular values to be considered non-zero.
    /// </summary>
    [Description("The absolute tolerance for singular values to be considered non-zero.")]
    public double? AbsoluteTolerance { get; set; }

    /// <summary>
    /// Gets or sets the relative tolerance for singular values to be considered non-zero.
    /// </summary>
    [Description("The relative tolerance for singular values to be considered non-zero.")]
    public double? RelativeTolerance { get; set; }

    /// <summary>
    /// Gets or sets whether to treat the input matrix as Hermitian if input is complex or symmetric if real.
    /// </summary>
    /// <remarks>
    /// True indicates that the input matrix is Hermitian if complex or symmetric if real; otherwise, the input matrix
    /// is treated as a generic matrix.
    /// </remarks>
    [Description("True indicates that the input matrix is Hermitian if complex or symmetric if real; otherwise, the input matrix is treated as a generic matrix.")]
    public bool Hermitian { get; set; }

    /// <summary>
    /// Computes the numerical rank of a matrix.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(tensor => matrix_rank(tensor, atol: AbsoluteTolerance, rtol: RelativeTolerance, hermitian: Hermitian));
    }
}
